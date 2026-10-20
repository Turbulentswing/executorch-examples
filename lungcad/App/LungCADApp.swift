import SwiftUI
import UniformTypeIdentifiers
import LungCADCore

@main
struct LungCADApp: App {
    var body: some Scene { WindowGroup { ContentView() } }
}

@MainActor
final class StudyModel: ObservableObject {
    @Published var volume: CTVolume?
    @Published var candidates: [Candidate] = []
    @Published var selected: UUID?
    @Published var slice = 0
    @Published var busy = false
    @Published var ready = false
    @Published var status = "Import one axial CT series to begin."
    @Published var progress = 0.0
    private let detector = Detector()
    private var operation: Task<Void, Never>?

    func importStudy(_ urls: [URL]) {
        guard !busy else { return }
        volume = nil; candidates = []; selected = nil; slice = 0
        busy = true; status = "Reading CT locally…"
        operation = Task {
            do {
                let worker = Task.detached(priority: .userInitiated) {
                    try Self.readStudy(urls)
                }
                let loaded = try await withTaskCancellationHandler {
                    try await worker.value
                } onCancel: { worker.cancel() }
                try Task.checkCancellation()
                volume = loaded; slice = loaded.size[2] / 2
                status = "CT loaded. Detection is experimental and not clinically validated."
            } catch is CancellationError { status = "Import cancelled."
            } catch { status = error.localizedDescription }
            busy = false
        }
    }

    nonisolated private static func readStudy(_ urls: [URL]) throws -> CTVolume {
        let scopes = urls.filter { $0.startAccessingSecurityScopedResource() }
        defer { scopes.forEach { $0.stopAccessingSecurityScopedResource() } }
        var files: [URL] = []
        for url in urls {
            let values = try url.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
            guard values.isSymbolicLink != true else { throw CADError.invalid("Symbolic links are unsupported.") }
            if values.isDirectory == true {
                guard let enumerator = FileManager.default.enumerator(
                    at: url, includingPropertiesForKeys: [.isRegularFileKey, .isSymbolicLinkKey],
                    options: [.skipsHiddenFiles]) else { throw CADError.invalid("Cannot read folder.") }
                for case let file as URL in enumerator {
                    try Task.checkCancellation()
                    let info = try file.resourceValues(forKeys: [.isRegularFileKey, .isSymbolicLinkKey])
                    if info.isSymbolicLink == true { throw CADError.invalid("Symbolic links are unsupported.") }
                    if info.isRegularFile == true { files.append(file) }
                    guard files.count <= 2048 else { throw CADError.invalid("Too many files. Select one CT series.") }
                }
            } else { files.append(url) }
        }
        guard (2...2048).contains(files.count) else { throw CADError.invalid("Select 2–2048 CT files from one series.") }
        var slices: [CTSlice] = []
        var voxels = 0
        for file in files {
            try Task.checkCancellation()
            let bytes = try file.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? Int.max
            guard bytes <= 64 * 1024 * 1024 else { throw CADError.invalid("A selected file exceeds 64 MiB.") }
            let slice = try DICOMReader.read(Data(contentsOf: file, options: .mappedIfSafe))
            voxels += slice.hu.count
            guard voxels <= 160_000_000 else { throw CADError.invalid("CT volume exceeds memory limit.") }
            slices.append(slice)
        }
        return try CTVolume(slices: slices)
    }

    func importModel(_ folder: URL) {
        guard !busy else { return }
        ready = false; busy = true; candidates = []; selected = nil
        operation = Task {
            let access = folder.startAccessingSecurityScopedResource()
            defer { if access { folder.stopAccessingSecurityScopedResource() }; busy = false }
            do {
                try await detector.load(folder: folder)
                ready = true
                status = "Model loaded. Synthetic parity only; iPad and clinical performance remain unverified."
            } catch { status = error.localizedDescription }
        }
    }

    func run() {
        guard !busy, ready, let volume else { return }
        candidates = []; selected = nil; busy = true; progress = 0
        status = "Experimental detection running offline…"
        operation = Task {
            do {
                let result = try await detector.detect(volume) { value in
                    await MainActor.run { self.progress = value }
                }
                try Task.checkCancellation()
                candidates = result
                status = "\(result.count) experimental candidates. A negative result does not exclude nodules."
            } catch is CancellationError { status = "Detection cancelled. No complete result."
            } catch { status = error.localizedDescription }
            busy = false
        }
    }

    func cancel() { operation?.cancel() }
    func clear() {
        guard !busy else { return }
        volume = nil; candidates = []; selected = nil; slice = 0
        status = "Study cleared from application memory."
    }
    func select(_ candidate: Candidate) {
        guard let volume else { return }
        selected = candidate.id
        slice = min(max(Int(volume.originalVoxel(ras: candidate.center)[2].rounded()), 0), volume.size[2] - 1)
    }
}

struct ContentView: View {
    @StateObject private var model = StudyModel()
    @State private var importCT = false
    @State private var importCTFiles = false
    @State private var importModel = false

    var body: some View {
        NavigationSplitView {
            List {
                Section("Local inputs") {
                    Button("Import CT folder", systemImage: "folder") { importCT = true }
                        .disabled(model.busy)
                        .fileImporter(isPresented: $importCT, allowedContentTypes: [.folder]) { result in
                            switch result {
                            case .success(let url): model.importStudy([url])
                            case .failure(let error): model.status = error.localizedDescription
                            }
                        }
                    Button("Import CT files", systemImage: "doc.on.doc") { importCTFiles = true }
                        .disabled(model.busy)
                        // DICOM exports often have unknown extensions or no extension.
                        // Accept items here; DICOMReader validates their actual contents.
                        .fileImporter(isPresented: $importCTFiles, allowedContentTypes: [.item],
                                      allowsMultipleSelection: true) { result in
                            switch result {
                            case .success(let urls): model.importStudy(urls)
                            case .failure(let error): model.status = error.localizedDescription
                            }
                        }
                    Button("Import model folder", systemImage: "cpu") { importModel = true }
                        .disabled(model.busy)
                        .fileImporter(isPresented: $importModel, allowedContentTypes: [.folder]) { result in
                            switch result {
                            case .success(let url): model.importModel(url)
                            case .failure(let error): model.status = error.localizedDescription
                            }
                        }
                    Button("Run experimental detection", systemImage: "viewfinder") { model.run() }
                        .disabled(model.busy || !model.ready || model.volume == nil)
                    if model.busy {
                        ProgressView(value: model.progress)
                        Button("Cancel", role: .cancel) { model.cancel() }
                    }
                    Button("Clear study", role: .destructive) { model.clear() }.disabled(model.busy)
                }
                Section("Status") {
                    Text(model.status).font(.callout).accessibilityIdentifier("status")
                    Text("Research prototype • not for diagnosis").font(.caption).foregroundStyle(.secondary)
                }
                Section("Candidates") {
                    ForEach(Array(model.candidates.enumerated()), id: \.element.id) { index, candidate in
                        Button {
                            model.select(candidate)
                        } label: {
                            VStack(alignment: .leading) {
                                Text("Candidate \(index + 1)")
                                Text(String(format: "%.1f mm box • score %.3f", candidate.diameter, candidate.score))
                                    .font(.caption).foregroundStyle(.secondary)
                            }
                        }
                    }
                }
            }
            .navigationTitle("LungCAD")
        } detail: {
            if let volume = model.volume {
                VStack {
                    SliceView(volume: volume, slice: model.slice,
                              candidates: model.candidates, selected: model.selected)
                    HStack {
                        Text("Slice \(model.slice + 1) / \(volume.size[2])")
                        Slider(value: Binding(get: { Double(model.slice) },
                                              set: { model.slice = Int($0) }),
                               in: 0...Double(volume.size[2] - 1), step: 1)
                    }.padding()
                }
            } else {
                ContentUnavailableView("Import a CT series", systemImage: "square.stack.3d.up",
                    description: Text("Uncompressed, single-frame axial DICOM CT. Processing stays on this iPad."))
            }
        }
    }
}

struct SliceView: View {
    let volume: CTVolume
    let slice: Int
    let candidates: [Candidate]
    let selected: UUID?

    var body: some View {
        GeometryReader { geometry in
            let aspect = Double(volume.size[0]) * volume.spacing[0] / (Double(volume.size[1]) * volume.spacing[1])
            let width = min(geometry.size.width, geometry.size.height * aspect)
            let height = width / aspect
            let left = (geometry.size.width - width) / 2
            let top = (geometry.size.height - height) / 2
            ZStack(alignment: .topLeading) {
                Color.black
                if let image = makeImage() {
                    Image(decorative: image, scale: 1).resizable().interpolation(.none)
                        .frame(width: width, height: height).offset(x: left, y: top)
                }
                ForEach(candidates) { candidate in
                    let a = volume.originalVoxel(ras: candidate.low)
                    let b = volume.originalVoxel(ras: candidate.high)
                    if Double(slice) >= a[2], Double(slice) <= b[2] {
                        Rectangle().stroke(candidate.id == selected ? Color.yellow : Color.green, lineWidth: 2)
                            .frame(width: abs(b[0] - a[0]) / Double(volume.size[0]) * width,
                                   height: abs(b[1] - a[1]) / Double(volume.size[1]) * height)
                            .offset(x: left + (min(a[0], b[0]) + 0.5) / Double(volume.size[0]) * width,
                                    y: top + (min(a[1], b[1]) + 0.5) / Double(volume.size[1]) * height)
                    }
                }
                Text("R").foregroundStyle(.white).offset(x: left + 8, y: top + height / 2)
                Text("L").foregroundStyle(.white).offset(x: left + width - 20, y: top + height / 2)
                Text("A").foregroundStyle(.white).offset(x: left + width / 2, y: top + 8)
            }
        }
    }

    private func makeImage() -> CGImage? {
        let count = volume.size[0] * volume.size[1]
        let start = slice * count
        // Fixed lung window: center -600 HU, width 1500 HU.
        let pixels = volume.hu[start..<start + count].map {
            UInt8(min(max(($0 + 1350) / 1500 * 255, 0), 255))
        }
        guard let provider = CGDataProvider(data: Data(pixels) as CFData) else { return nil }
        return CGImage(width: volume.size[0], height: volume.size[1], bitsPerComponent: 8,
                       bitsPerPixel: 8, bytesPerRow: volume.size[0], space: CGColorSpaceCreateDeviceGray(),
                       bitmapInfo: CGBitmapInfo(rawValue: 0), provider: provider,
                       decode: nil, shouldInterpolate: false, intent: .defaultIntent)
    }
}
