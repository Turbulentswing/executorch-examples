// swift-tools-version: 5.9
import PackageDescription

let package = Package(
    name: "LungCADCore",
    platforms: [.macOS(.v13), .iOS(.v17)],
    products: [.library(name: "LungCADCore", targets: ["LungCADCore"])],
    targets: [
        .target(name: "LungCADCore"),
        .testTarget(name: "LungCADCoreTests", dependencies: ["LungCADCore"],
                    resources: [.process("Resources")])
    ]
)
