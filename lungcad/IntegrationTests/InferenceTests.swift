import XCTest
import Foundation
import OnnxRuntimeBindings
@testable import LungCAD
@testable import LungCADCore

final class InferenceTests: XCTestCase {
    private func folder() throws -> URL {
        try XCTUnwrap(Bundle(for: Self.self).url(forResource: "Model", withExtension: nil),
                      "Run the model workflow to stage the synthetic model fixture.")
    }

    func testNativeONNXMatchesPyTorchUniformTensor() throws {
        let folder = try folder()
        let env = try ORTEnv(loggingLevel: .warning)
        let options = try ORTSessionOptions()
        try options.setIntraOpNumThreads(2)
        let session = try ORTSession(env: env, modelPath: folder.appendingPathComponent("lungcad.onnx").path,
                                     sessionOptions: options)
        let values = [Float](repeating: 0.5, count: 96 * 96 * 64)
        let data = values.withUnsafeBytes { NSMutableData(bytes: $0.baseAddress!, length: $0.count) }
        let tensor = try ORTValue(tensorData: data, elementType: .float, shape: [1, 1, 96, 96, 64])
        let outputs = try session.run(withInputs: ["image": tensor],
                                      outputNames: ["boxes", "scores"], runOptions: nil)
        for name in ["boxes", "scores"] {
            let actual = try XCTUnwrap(outputs[name]).tensorData() as Data
            let expected = try Data(contentsOf: folder.appendingPathComponent("uniform-\(name).bin"))
            XCTAssertEqual(actual.count, expected.count)
            guard actual.count == expected.count else { return }
            var worstRatio: Float = 0
            actual.withUnsafeBytes { a in expected.withUnsafeBytes { e in
                for i in stride(from: 0, to: a.count, by: 4) {
                    let av = a.loadUnaligned(fromByteOffset: i, as: Float.self)
                    let ev = e.loadUnaligned(fromByteOffset: i, as: Float.self)
                    if !av.isFinite { worstRatio = .infinity; break }
                    worstRatio = max(worstRatio, abs(av - ev) / (0.002 + 0.002 * abs(ev)))
                }
            } }
            XCTAssertLessThanOrEqual(worstRatio, 1, "\(name) exceeded PyTorch parity tolerance")
        }
    }

    func testNativePipelineMatchesMONAIUniformCandidates() async throws {
        struct Expected: Decodable { let boxes: [[Double]]; let scores: [Float] }
        let folder = try folder()
        let expected = try JSONDecoder().decode(Expected.self,
            from: Data(contentsOf: folder.appendingPathComponent("uniform-candidates.json")))
        let slices = (0..<64).map { z in
            CTSlice(series: "synthetic", instance: "\(z)", rows: 96, columns: 96,
                    position: [0, 0, Double(z) * 1.25], orientation: [1, 0, 0, 0, 1, 0],
                    spacing: [0.703125, 0.703125],
                    hu: [Float](repeating: -362, count: 96 * 96))
        }
        let volume = try CTVolume(slices: slices)
        let detector = Detector()
        try await detector.load(folder: folder)
        let candidates = try await detector.detect(volume) { _ in }
        XCTAssertEqual(candidates.count, expected.boxes.count)
        for (candidate, pair) in zip(candidates, zip(expected.boxes, expected.scores)) {
            XCTAssertEqual(candidate.score, pair.1, accuracy: 0.002)
            for axis in 0..<3 {
                XCTAssertEqual(candidate.low[axis],
                    volume.rasOrigin[axis] + pair.0[axis] * Preprocessor.spacing[axis], accuracy: 0.02)
                XCTAssertEqual(candidate.high[axis],
                    volume.rasOrigin[axis] + pair.0[axis + 3] * Preprocessor.spacing[axis], accuracy: 0.02)
            }
        }
    }
}
