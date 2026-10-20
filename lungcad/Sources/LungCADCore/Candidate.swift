import Foundation

public struct Candidate: Identifiable, Sendable {
    public let id: UUID
    public let low: [Double] // RAS millimetres
    public let high: [Double]
    public let score: Float
    public init(low: [Double], high: [Double], score: Float) {
        id = UUID(); self.low = low; self.high = high; self.score = score
    }
    public var center: [Double] { (0..<3).map { (low[$0] + high[$0]) / 2 } }
    public var diameter: Double { (0..<3).map { high[$0] - low[$0] }.max() ?? 0 }
}

public enum CandidateFilter {
    public static func iou(_ a: Candidate, _ b: Candidate) -> Double {
        let intersection = (0..<3).map { max(0, min(a.high[$0], b.high[$0]) - max(a.low[$0], b.low[$0])) }.reduce(1, *)
        let av = (0..<3).map { max(0, a.high[$0] - a.low[$0]) }.reduce(1, *)
        let bv = (0..<3).map { max(0, b.high[$0] - b.low[$0]) }.reduce(1, *)
        return intersection / max(av + bv - intersection, 1e-12)
    }
    public static func nms(_ input: [Candidate], threshold: Double = 0.22, limit: Int = 300) -> [Candidate] {
        let valid = input.filter { c in
            c.low.count == 3 && c.high.count == 3 && c.score.isFinite &&
            c.low.allSatisfy(\.isFinite) && c.high.allSatisfy(\.isFinite) &&
            (0..<3).allSatisfy { c.high[$0] > c.low[$0] }
        }
        var kept: [Candidate] = []
        for c in valid.sorted(by: { $0.score > $1.score }) {
            if !kept.contains(where: { iou(c, $0) > threshold }) { kept.append(c) }
            if kept.count >= limit { break }
        }
        return kept
    }
}
