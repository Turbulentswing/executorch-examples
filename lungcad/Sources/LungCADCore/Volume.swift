import Foundation

public struct CTVolume: Sendable {
    public let size: [Int] // original column, row, slice
    public let spacing: [Double]
    public let originLPS: [Double]
    public let hu: [Float] // column fastest

    /// Initial supported geometry is axial, increasing L/P/S. Oblique, tilted,
    /// flipped and irregular acquisitions are rejected explicitly.
    public init(slices: [CTSlice]) throws {
        guard slices.count >= 2, slices.count <= 2048, let first = slices.first else {
            throw CADError.invalid("Select 2–2048 CT slices from one series.")
        }
        let axial: [Double] = [1, 0, 0, 0, 1, 0]
        guard Set(slices.map(\.instance)).count == slices.count else { throw CADError.invalid("Duplicate slices.") }
        for slice in slices {
            guard slice.series == first.series, slice.rows == first.rows,
                  slice.columns == first.columns,
                  zip(slice.spacing, first.spacing).allSatisfy({ abs($0 - $1) < 1e-5 }),
                  zip(slice.orientation, axial).allSatisfy({ abs($0 - $1) < 1e-5 }),
                  abs(slice.position[0] - first.position[0]) < 0.01,
                  abs(slice.position[1] - first.position[1]) < 0.01 else {
                throw CADError.invalid("Select one axial CT series. Mixed, oblique or tilted geometry is unsupported.")
            }
        }
        let sorted = slices.sorted { $0.position[2] < $1.position[2] }
        let dz = sorted[1].position[2] - sorted[0].position[2]
        guard dz > 0.01, zip(sorted.dropFirst(), sorted).allSatisfy({
            abs(($0.position[2] - $1.position[2]) - dz) < max(0.01, dz * 0.01)
        }) else { throw CADError.invalid("Duplicate positions, missing slices or irregular slice spacing.") }
        guard first.rows * first.columns * slices.count <= 160_000_000 else {
            throw CADError.invalid("CT volume exceeds the prototype memory limit.")
        }
        size = [first.columns, first.rows, sorted.count]
        spacing = [first.spacing[1], first.spacing[0], dz]
        originLPS = sorted[0].position
        hu = sorted.flatMap(\.hu)
    }

    public func value(x: Int, y: Int, z: Int) -> Float {
        hu[(z * size[1] + y) * size[0] + x]
    }

    public var rasOrigin: [Double] {
        [-originLPS[0] - Double(size[0] - 1) * spacing[0],
         -originLPS[1] - Double(size[1] - 1) * spacing[1], originLPS[2]]
    }

    public func originalVoxel(ras: [Double]) -> [Double] {
        [(-ras[0] - originLPS[0]) / spacing[0],
         (-ras[1] - originLPS[1]) / spacing[1],
         (ras[2] - originLPS[2]) / spacing[2]]
    }
}

public enum Preprocessor {
    public static let spacing = [0.703125, 0.703125, 1.25]
    public static let patch = [96, 96, 64]

    public static func resampledSize(_ volume: CTVolume) -> [Int] {
        (0..<3).map { Int((Double(volume.size[$0] - 1) * volume.spacing[$0] / spacing[$0]).rounded()) + 1 }
    }

    public static func starts(length: Int, patch: Int) -> [Int] {
        guard length > patch else { return [0] }
        let step = patch * 3 / 4
        var result = Array(stride(from: 0, through: length - patch, by: step))
        if result.last != length - patch { result.append(length - patch) }
        return result
    }

    /// Samples a RAS patch directly from the source volume, avoiding a second
    /// full-volume allocation. Tensor order is N,C,X,Y,Z (Z contiguous).
    public static func tensor(_ volume: CTVolume, start: [Int]) throws -> [Float] {
        let dimensions = resampledSize(volume)
        var result = [Float](repeating: 0, count: patch.reduce(1, *))
        for x in 0..<patch[0] {
            try Task.checkCancellation()
            for y in 0..<patch[1] {
                for z in 0..<patch[2] {
                    let v = [x + start[0], y + start[1], z + start[2]]
                    if (0..<3).contains(where: { v[$0] >= dimensions[$0] }) { continue }
                    let p = [
                        Double(volume.size[0] - 1) - Double(v[0]) * spacing[0] / volume.spacing[0],
                        Double(volume.size[1] - 1) - Double(v[1]) * spacing[1] / volume.spacing[1],
                        Double(v[2]) * spacing[2] / volume.spacing[2]
                    ]
                    let clamped = (0..<3).map { min(max(p[$0], 0), Double(volume.size[$0] - 1)) }
                    let lo = clamped.map { Int(floor($0)) }
                    let hi = (0..<3).map { min(lo[$0] + 1, volume.size[$0] - 1) }
                    let f = (0..<3).map { Float(clamped[$0] - Double(lo[$0])) }
                    var hu: Float = 0
                    for a in 0...1 { for b in 0...1 { for c in 0...1 {
                        hu += volume.value(x: a == 0 ? lo[0] : hi[0],
                                           y: b == 0 ? lo[1] : hi[1],
                                           z: c == 0 ? lo[2] : hi[2])
                            * (a == 0 ? 1 - f[0] : f[0])
                            * (b == 0 ? 1 - f[1] : f[1])
                            * (c == 0 ? 1 - f[2] : f[2])
                    } } }
                    result[(x * patch[1] + y) * patch[2] + z] = min(max((hu + 1024) / 1324, 0), 1)
                }
            }
        }
        return result
    }
}
