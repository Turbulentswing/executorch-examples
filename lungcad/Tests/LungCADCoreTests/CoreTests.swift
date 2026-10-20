import Foundation
import XCTest
@testable import LungCADCore

final class CoreTests: XCTestCase {
    func testMONAIPreprocessingReference() throws {
        struct Reference: Decodable {
            let sourceSize: [Int]
            let sourceSpacing: [Double]
            let sourceOriginLPS: [Double]
            let sourceHU: [Float]
            let expectedSize: [Int]
            let expectedRASOrigin: [Double]
            let expectedTensor: [Float]
        }
        let url = try XCTUnwrap(Bundle.module.url(forResource: "preprocessing", withExtension: "json"))
        let ref = try JSONDecoder().decode(Reference.self, from: Data(contentsOf: url))
        let n = ref.sourceSize[0] * ref.sourceSize[1]
        let slices = (0..<ref.sourceSize[2]).map { z in
            CTSlice(series: "synthetic", instance: "\(z)", rows: ref.sourceSize[1],
                    columns: ref.sourceSize[0],
                    position: [ref.sourceOriginLPS[0], ref.sourceOriginLPS[1],
                               ref.sourceOriginLPS[2] + Double(z) * ref.sourceSpacing[2]],
                    orientation: [1, 0, 0, 0, 1, 0],
                    spacing: [ref.sourceSpacing[1], ref.sourceSpacing[0]],
                    hu: Array(ref.sourceHU[z * n..<(z + 1) * n]))
        }
        let volume = try CTVolume(slices: slices)
        XCTAssertEqual(Preprocessor.resampledSize(volume), ref.expectedSize)
        for axis in 0..<3 {
            XCTAssertEqual(volume.rasOrigin[axis], ref.expectedRASOrigin[axis], accuracy: 1e-6)
        }
        let tensor = try Preprocessor.tensor(volume, start: [0, 0, 0])
        for x in 0..<ref.expectedSize[0] { for y in 0..<ref.expectedSize[1] { for z in 0..<ref.expectedSize[2] {
            let i = (x * Preprocessor.patch[1] + y) * Preprocessor.patch[2] + z
            let j = (x * ref.expectedSize[1] + y) * ref.expectedSize[2] + z
            XCTAssertEqual(tensor[i], ref.expectedTensor[j], accuracy: 1e-5)
        } } }
    }

    func testSignedPixelsRescaleAndImplicitVR() throws {
        for explicit in [true, false] {
            let slice = try DICOMReader.read(fixture(explicit: explicit, pixels: [0xFFF, 0, 100, 0x800]))
            XCTAssertEqual(slice.hu, [-1026, -1024, -824, -5120])
            XCTAssertEqual(slice.spacing, [1, 1])
        }
    }

    func testRejectTruncatedCompressedAndMalformedFiles() {
        XCTAssertThrowsError(try DICOMReader.read(Data()))
        var truncated = fixture()
        truncated.removeLast()
        XCTAssertThrowsError(try DICOMReader.read(truncated))
        XCTAssertThrowsError(try DICOMReader.read(fixture(syntax: "1.2.840.10008.1.2.4.90")))
        XCTAssertThrowsError(try DICOMReader.read(fixture(bits: 17)))
        XCTAssertThrowsError(try DICOMReader.read(fixture(spacing: "0\\1")))
        XCTAssertThrowsError(try DICOMReader.read(fixture(slope: "nan")))
    }

    func testSortingAndCoordinateRoundTrip() throws {
        let low = try DICOMReader.read(fixture(z: 10, instance: "1"))
        let high = try DICOMReader.read(fixture(z: 12, instance: "2"))
        let volume = try CTVolume(slices: [high, low])
        XCTAssertEqual(volume.size, [2, 2, 2])
        XCTAssertEqual(volume.spacing, [1, 1, 2])
        XCTAssertEqual(volume.originLPS, [0, 0, 10])
        XCTAssertEqual(volume.rasOrigin, [-1, -1, 10])
        XCTAssertEqual(volume.originalVoxel(ras: volume.rasOrigin), [1, 1, 0])
        XCTAssertEqual(volume.originalVoxel(ras: [0, 0, 12]), [0, 0, 1])
    }

    func testRejectMixedDuplicateObliqueAndMissingSlices() throws {
        let first = try DICOMReader.read(fixture(z: 0, instance: "1"))
        let second = try DICOMReader.read(fixture(z: 1, instance: "2"))
        XCTAssertThrowsError(try CTVolume(slices: [first, first]))
        XCTAssertThrowsError(try CTVolume(slices: [
            first, DICOMReader.read(fixture(z: 1, instance: "2", series: "other"))
        ]))
        XCTAssertThrowsError(try CTVolume(slices: [
            first, DICOMReader.read(fixture(z: 1, instance: "2", orientation: "0\\1\\0\\1\\0\\0"))
        ]))
        XCTAssertThrowsError(try CTVolume(slices: [
            first, second, DICOMReader.read(fixture(z: 3, instance: "3"))
        ]))
        XCTAssertThrowsError(try CTVolume(slices: [
            first, DICOMReader.read(fixture(z: 0, instance: "2"))
        ]))
    }

    func testRASResamplingIntensityAndPadding() throws {
        let slices = try [0, 1].map {
            try DICOMReader.read(fixture(z: Double($0), instance: "\($0)",
                                         pixels: [0, 100, 200, 300]))
        }
        let volume = try CTVolume(slices: slices)
        let tensor = try Preprocessor.tensor(volume, start: [0, 0, 0])
        // RAS origin is the original last row/column: HU = 300*2-1024.
        XCTAssertEqual(tensor[0], 600 / 1324, accuracy: 1e-6)
        // One RAS X step interpolates from pixel 300 toward 200.
        XCTAssertEqual(tensor[96 * 64], Float((600 - 200 * 0.703125) / 1324), accuracy: 1e-6)
        XCTAssertEqual(tensor[63], 0) // outside the resampled volume -> normalized air
        XCTAssertEqual(tensor.count, 96 * 96 * 64)
    }

    func testPatchCoverage() {
        XCTAssertEqual(Preprocessor.starts(length: 20, patch: 96), [0])
        let starts = Preprocessor.starts(length: 251, patch: 96)
        XCTAssertEqual(starts.first, 0)
        XCTAssertEqual(starts.last, 155)
        for i in 0..<251 {
            XCTAssertTrue(starts.contains { $0 <= i && i < $0 + 96 })
        }
    }

    func testNMSUsesThreeDimensionsAndRejectsInvalidBoxes() {
        let a = Candidate(low: [0, 0, 0], high: [10, 10, 10], score: 0.9)
        let duplicate = Candidate(low: [0, 0, 0], high: [10, 10, 10], score: 0.8)
        let distant = Candidate(low: [0, 0, 11], high: [10, 10, 21], score: 0.7)
        let invalid = Candidate(low: [0, 0, 0], high: [-1, 10, 10], score: 1)
        let output = CandidateFilter.nms([duplicate, distant, invalid, a])
        XCTAssertEqual(output.map(\.id), [a.id, distant.id])
        XCTAssertEqual(CandidateFilter.iou(a, duplicate), 1)
        XCTAssertEqual(CandidateFilter.iou(a, distant), 0)
    }

    private func fixture(explicit: Bool = true, z: Double = 0, instance: String = "1",
                         series: String = "1.2.3", pixels: [UInt16] = [0, 0, 0, 0],
                         syntax: String? = nil, bits: UInt16 = 12, spacing: String = "1\\1",
                         slope: String = "2", orientation: String = "1\\0\\0\\0\\1\\0") -> Data {
        var result = Data(repeating: 0, count: 128)
        result.append(Data("DICM".utf8))
        func le16(_ value: UInt16) -> Data { Data([UInt8(value & 255), UInt8(value >> 8)]) }
        func le32(_ value: UInt32) -> Data {
            Data((0..<4).map { UInt8((value >> ($0 * 8)) & 255) })
        }
        func tag(_ group: UInt16, _ element: UInt16, _ vr: String, _ value: Data, meta: Bool = false) {
            var value = value
            if value.count % 2 != 0 { value.append(vr == "UI" ? 0 : 32) }
            result.append(le16(group)); result.append(le16(element))
            if explicit || meta {
                result.append(Data(vr.utf8))
                if vr == "OW" {
                    result.append(Data([0, 0])); result.append(le32(UInt32(value.count)))
                } else { result.append(le16(UInt16(value.count))) }
            } else { result.append(le32(UInt32(value.count))) }
            result.append(value)
        }
        func text(_ g: UInt16, _ e: UInt16, _ vr: String, _ value: String, meta: Bool = false) {
            tag(g, e, vr, Data(value.utf8), meta: meta)
        }
        text(2, 0x10, "UI", syntax ?? (explicit ? "1.2.840.10008.1.2.1" : "1.2.840.10008.1.2"), meta: true)
        text(8, 0x16, "UI", "1.2.840.10008.5.1.4.1.1.2")
        text(8, 0x18, "UI", instance)
        text(8, 0x60, "CS", "CT")
        text(0x20, 0xE, "UI", series)
        text(0x20, 0x32, "DS", "0\\0\\\(z)")
        text(0x20, 0x37, "DS", orientation)
        tag(0x28, 2, "US", le16(1))
        text(0x28, 4, "CS", "MONOCHROME2")
        tag(0x28, 0x10, "US", le16(2))
        tag(0x28, 0x11, "US", le16(2))
        text(0x28, 0x30, "DS", spacing)
        tag(0x28, 0x100, "US", le16(16))
        tag(0x28, 0x101, "US", le16(bits))
        tag(0x28, 0x102, "US", le16(bits - 1))
        tag(0x28, 0x103, "US", le16(1))
        text(0x28, 0x1052, "DS", "-1024")
        text(0x28, 0x1053, "DS", slope)
        tag(0x7FE0, 0x10, "OW", pixels.reduce(into: Data()) { $0.append(le16($1)) })
        return result
    }
}
