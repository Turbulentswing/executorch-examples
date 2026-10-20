import Foundation

public enum CADError: Error, LocalizedError {
    case invalid(String)
    public var errorDescription: String? {
        switch self { case .invalid(let message): return message }
    }
}

public struct CTSlice: Sendable {
    public let series: String
    public let instance: String
    public let rows: Int
    public let columns: Int
    public let position: [Double]
    public let orientation: [Double]
    public let spacing: [Double]
    public let hu: [Float]
}

/// Deliberately bounded Part 10 reader. Supports uncompressed little-endian
/// classic CT only; unsupported encodings fail instead of producing guessed pixels.
public enum DICOMReader {
    public static func read(_ data: Data) throws -> CTSlice {
        guard data.count <= 64 * 1024 * 1024, data.count >= 132,
              data[128..<132] == Data("DICM".utf8) else {
            throw CADError.invalid("Expected a DICOM Part 10 file under 64 MiB.")
        }
        var offset = 132
        var explicit = true
        var syntax: String?
        var fields: [UInt32: Data] = [:]
        let wanted: Set<UInt32> = [
            0x00080016, 0x00080018, 0x00080060, 0x00080008,
            0x0020000E, 0x00200032, 0x00200037, 0x00280002,
            0x00280004, 0x00280008, 0x00280010, 0x00280011,
            0x00280030, 0x00280100, 0x00280101, 0x00280102,
            0x00280103, 0x00281052, 0x00281053, 0x7FE00010
        ]
        func u16(_ p: Int) -> Int { Int(data[p]) | Int(data[p + 1]) << 8 }
        func u32(_ p: Int) -> Int { u16(p) | u16(p + 2) << 16 }
        let longVR: Set<String> = ["OB", "OD", "OF", "OL", "OV", "OW", "SQ", "UC", "UR", "UT", "UN"]
        while offset < data.count {
            guard offset + 8 <= data.count else { throw CADError.invalid("Truncated DICOM header.") }
            let group = u16(offset), element = u16(offset + 2)
            if group != 2 {
                guard let syntax else { throw CADError.invalid("Missing transfer syntax.") }
                guard ["1.2.840.10008.1.2", "1.2.840.10008.1.2.1"].contains(syntax) else {
                    throw CADError.invalid("Compressed or big-endian DICOM is not supported. Export uncompressed CT.")
                }
                explicit = syntax != "1.2.840.10008.1.2"
            }
            let tag = UInt32(group << 16 | element)
            var length: Int
            if explicit || group == 2 {
                let vr = String(data: data[offset + 4..<offset + 6], encoding: .ascii) ?? ""
                guard vr.count == 2, vr.allSatisfy({ $0.isUppercase }) else {
                    throw CADError.invalid("Invalid DICOM value representation.")
                }
                if longVR.contains(vr) {
                    guard offset + 12 <= data.count else { throw CADError.invalid("Truncated DICOM length.") }
                    length = u32(offset + 8)
                    offset += 12
                } else {
                    length = u16(offset + 6)
                    offset += 8
                }
            } else {
                length = u32(offset + 4)
                offset += 8
            }
            guard length != 0xFFFFFFFF else {
                throw CADError.invalid("Undefined-length sequences are not supported in this importer.")
            }
            guard length <= data.count - offset else { throw CADError.invalid("Truncated DICOM value.") }
            if tag == 0x00020010 { syntax = string(Data(data[offset..<offset + length])) }
            if wanted.contains(tag) {
                guard fields[tag] == nil else { throw CADError.invalid("Duplicate required DICOM tag.") }
                fields[tag] = Data(data[offset..<offset + length])
            }
            offset += length
            if tag == 0x7FE00010 { break }
        }
        func text(_ tag: UInt32) -> String { string(fields[tag] ?? Data()) }
        func number(_ tag: UInt32) throws -> Int {
            guard let d = fields[tag], d.count == 2 else { throw CADError.invalid("Missing CT pixel metadata.") }
            return Int(d[0]) | Int(d[1]) << 8
        }
        func decimals(_ tag: UInt32, count: Int) throws -> [Double] {
            let parts = text(tag).split(separator: "\\", omittingEmptySubsequences: false)
            let values = parts.compactMap { Double($0.trimmingCharacters(in: .whitespaces)) }
            guard values.count == count, parts.count == count, values.allSatisfy(\.isFinite) else {
                throw CADError.invalid("Missing or invalid CT geometry.")
            }
            return values
        }
        guard text(0x00080016) == "1.2.840.10008.5.1.4.1.1.2",
              text(0x00080060) == "CT",
              !text(0x00080008).uppercased().contains("LOCALIZER"),
              !text(0x0020000E).isEmpty, !text(0x00080018).isEmpty,
              ["", "1"].contains(text(0x00280008)),
              try number(0x00280002) == 1,
              text(0x00280004) == "MONOCHROME2",
              try number(0x00280100) == 16 else {
            throw CADError.invalid("Only single-frame, 16-bit MONOCHROME2 classic CT is supported.")
        }
        let rows = try number(0x00280010), columns = try number(0x00280011)
        let bits = try number(0x00280101), high = try number(0x00280102)
        let signed = try number(0x00280103)
        guard rows > 0, columns > 0, rows <= 2048, columns <= 2048,
              (1...16).contains(bits), high == bits - 1, (0...1).contains(signed),
              let pixels = fields[0x7FE00010], pixels.count == rows * columns * 2,
              let slope = Double(text(0x00281053)), slope.isFinite, slope != 0,
              let intercept = Double(text(0x00281052)), intercept.isFinite else {
            throw CADError.invalid("Invalid pixel data or HU rescale metadata.")
        }
        var hu = [Float]()
        hu.reserveCapacity(rows * columns)
        for p in stride(from: 0, to: pixels.count, by: 2) {
            var value = (Int(pixels[p]) | Int(pixels[p + 1]) << 8) & ((1 << bits) - 1)
            if signed == 1, value & (1 << (bits - 1)) != 0 { value -= 1 << bits }
            let result = Float(Double(value) * slope + intercept)
            guard result.isFinite else { throw CADError.invalid("Nonfinite HU value.") }
            hu.append(result)
        }
        let spacing = try decimals(0x00280030, count: 2)
        guard spacing.allSatisfy({ $0 > 0 }) else { throw CADError.invalid("Invalid pixel spacing.") }
        return CTSlice(series: text(0x0020000E), instance: text(0x00080018),
                       rows: rows, columns: columns,
                       position: try decimals(0x00200032, count: 3),
                       orientation: try decimals(0x00200037, count: 6),
                       spacing: spacing, hu: hu)
    }

    private static func string(_ data: Data) -> String {
        String(data: data, encoding: .ascii)?.trimmingCharacters(in: .whitespacesAndNewlines.union(.controlCharacters)) ?? ""
    }
}
