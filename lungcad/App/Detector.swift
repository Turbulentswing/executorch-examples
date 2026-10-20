import Foundation
import CryptoKit
import OnnxRuntimeBindings
import LungCADCore

struct ModelManifest: Decodable {
    let schema: Int
    let patch: [Int]
    let spacing: [Double]
    let levelCounts: [Int]
    let sha256: String
    let syntheticParityPassed: Bool
}

actor Detector {
    private var environment: ORTEnv?
    private var session: ORTSession?
    private var manifest: ModelManifest?

    func load(folder: URL) throws {
        session = nil; manifest = nil; environment = nil
        let config = try JSONDecoder().decode(ModelManifest.self,
            from: Data(contentsOf: folder.appendingPathComponent("lungcad.json")))
        guard config.schema == 1, config.patch == Preprocessor.patch,
              config.spacing == Preprocessor.spacing, config.syntheticParityPassed,
              config.levelCounts.count == 3,
              config.levelCounts.allSatisfy({ $0 > 0 && $0 < 1_000_000 }) else {
            throw CADError.invalid("Incompatible or unverified model export.")
        }
        let url = folder.appendingPathComponent("lungcad.onnx")
        let file = try FileHandle(forReadingFrom: url)
        defer { try? file.close() }
        var hasher = SHA256()
        while let block = try file.read(upToCount: 1024 * 1024), !block.isEmpty { hasher.update(data: block) }
        let hash = hasher.finalize().map { String(format: "%02x", $0) }.joined()
        guard hash == config.sha256 else { throw CADError.invalid("Model checksum mismatch.") }
        let env = try ORTEnv(loggingLevel: .warning)
        let options = try ORTSessionOptions()
        try options.setIntraOpNumThreads(4)
        let loaded = try ORTSession(env: env, modelPath: url.path, sessionOptions: options)
        guard Set(try loaded.inputNames()) == ["image"],
              Set(try loaded.outputNames()) == ["boxes", "scores"] else {
            throw CADError.invalid("Unexpected model inputs or outputs.")
        }
        environment = env; session = loaded; manifest = config
    }

    func detect(_ volume: CTVolume, progress: @Sendable (Double) async -> Void) async throws -> [Candidate] {
        guard let session, let manifest else { throw CADError.invalid("Import a validated model export first.") }
        let size = Preprocessor.resampledSize(volume)
        let axes = (0..<3).map { Preprocessor.starts(length: size[$0], patch: Preprocessor.patch[$0]) }
        let total = axes.map(\.count).reduce(1, *)
        var done = 0
        var candidates: [Candidate] = []
        for x in axes[0] { for y in axes[1] { for z in axes[2] {
            try Task.checkCancellation()
            let start = [x, y, z]
            let tensor = try Preprocessor.tensor(volume, start: start)
            let bytes = tensor.withUnsafeBytes { NSMutableData(bytes: $0.baseAddress!, length: $0.count) }
            let input = try ORTValue(tensorData: bytes, elementType: .float,
                                    shape: ([1, 1] + Preprocessor.patch).map { NSNumber(value: $0) })
            let outputs = try session.run(withInputs: ["image": input],
                                          outputNames: ["boxes", "scores"], runOptions: nil)
            guard let boxesOutput = outputs["boxes"], let scoresOutput = outputs["scores"] else {
                throw CADError.invalid("Missing detector output.")
            }
            let count = manifest.levelCounts.reduce(0, +)
            let boxInfo = try boxesOutput.tensorTypeAndShapeInfo()
            let scoreInfo = try scoresOutput.tensorTypeAndShapeInfo()
            guard boxInfo.elementType == .float, scoreInfo.elementType == .float,
                  boxInfo.shape.map(\.intValue) == [count, 6],
                  scoreInfo.shape.map(\.intValue) == [count] else {
                throw CADError.invalid("Invalid detector output shape.")
            }
            let boxes = try floats(boxesOutput)
            let scores = try floats(scoresOutput)
            guard boxes.count == count * 6, scores.count == count,
                  boxes.allSatisfy(\.isFinite), scores.allSatisfy(\.isFinite) else {
                throw CADError.invalid("Invalid or nonfinite model outputs.")
            }
            var offset = 0
            for levelCount in manifest.levelCounts {
                let selected = (offset..<offset + levelCount).filter { scores[$0] > 0.02 }
                    .sorted { scores[$0] > scores[$1] }.prefix(1000)
                for i in selected {
                    let lo = (0..<3).map { max(0, Double(boxes[i * 6 + $0])) }
                    let hi = (0..<3).map { min(Double(Preprocessor.patch[$0]), Double(boxes[i * 6 + 3 + $0])) }
                    let lowVoxel = (0..<3).map { min(Double(size[$0]), lo[$0] + Double(start[$0])) }
                    let highVoxel = (0..<3).map { min(Double(size[$0]), hi[$0] + Double(start[$0])) }
                    guard (0..<3).allSatisfy({ hi[$0] > lo[$0] && highVoxel[$0] > lowVoxel[$0] }) else { continue }
                    let low = (0..<3).map { volume.rasOrigin[$0] + lowVoxel[$0] * Preprocessor.spacing[$0] }
                    let high = (0..<3).map { volume.rasOrigin[$0] + highVoxel[$0] * Preprocessor.spacing[$0] }
                    candidates.append(Candidate(low: low, high: high, score: scores[i]))
                }
                offset += levelCount
            }
            // Explicit resource cap: do not silently discard potentially relevant candidates.
            guard candidates.count <= 100_000 else {
                throw CADError.invalid("Candidate memory limit reached. This series cannot be completed by the prototype.")
            }
            done += 1
            await progress(Double(done) / Double(total))
        } } }
        return CandidateFilter.nms(candidates)
    }

    private func floats(_ value: ORTValue) throws -> [Float] {
        let data = try value.tensorData() as Data
        guard data.count % 4 == 0 else { throw CADError.invalid("Invalid float tensor.") }
        return data.withUnsafeBytes { bytes in
            stride(from: 0, to: bytes.count, by: 4).map { bytes.loadUnaligned(fromByteOffset: $0, as: Float.self) }
        }
    }
}
