import Foundation

/// Errors thrown by the AthLink CV client.
public enum AthLinkError: Error, LocalizedError, Sendable {
    case invalidURL(String)
    case httpError(statusCode: Int, detail: String)
    case decodingFailed(Error)
    case timeout
    case jobFailed(String)

    public var errorDescription: String? {
        switch self {
        case .invalidURL(let url):       return "Invalid URL: \(url)"
        case .httpError(let code, let d): return "HTTP \(code): \(d)"
        case .decodingFailed(let err):    return "Decode error: \(err.localizedDescription)"
        case .timeout:                    return "Job polling timed out"
        case .jobFailed(let msg):         return "Job failed: \(msg)"
        }
    }
}

/// Swift client for the AthLink CV Service REST API.
///
/// Usage:
/// ```swift
/// let client = AthLinkClient(baseURL: "http://192.168.1.42:8001")
///
/// // 1. Track players (async job)
/// let queuedJob = try await client.trackPlayersWithTeams(
///     TrackPlayersRequest(jobId: "game1", videoPath: "/path/to/video.mp4")
/// )
///
/// // 2. Poll until done
/// let finalStatus = try await client.pollUntilDone(jobId: queuedJob.jobId)
///
/// // 3. Export results
/// let export = try await client.export(jobId: "game1")
/// print(export.tactics.team0Formation ?? "unknown")
/// ```
public final class AthLinkClient: Sendable {

    private let baseURL: String
    private let session: URLSession
    private let prefix: String

    /// Create a new client.
    /// - Parameters:
    ///   - baseURL: Scheme + host + port, e.g. `"http://192.168.1.42:8001"`
    ///   - session: Optional custom URLSession (defaults to `.shared`)
    public init(baseURL: String, session: URLSession = .shared) {
        self.baseURL = baseURL.hasSuffix("/") ? String(baseURL.dropLast()) : baseURL
        self.session = session
        self.prefix = "\(self.baseURL)/api/v1"
    }

    // MARK: - Health

    public func health() async throws -> HealthResponse {
        return try await get("\(prefix)/health")
    }

    // MARK: - Tracking

    public func trackPlayersWithTeams(_ req: TrackPlayersRequest) async throws -> QueuedJobResponse {
        return try await post("\(prefix)/track/players-with-teams", body: req)
    }

    public func trackPlayers(_ req: TrackPlayersRequest) async throws -> TrackPlayersResponse {
        return try await post("\(prefix)/track/players", body: req)
    }

    // MARK: - Pitch

    public func pitchMap(_ req: PitchMapRequest) async throws -> QueuedJobResponse {
        return try await post("\(prefix)/pitch/map", body: req)
    }

    // MARK: - Tactics

    public func tacticsAnalyze(_ req: TacticsRequest) async throws -> QueuedJobResponse {
        return try await post("\(prefix)/tactics/analyze", body: req)
    }

    // MARK: - Render

    public func render(jobId: String, includeMinimap: Bool = false) async throws -> QueuedJobResponse {
        let req = RenderRequest(jobId: jobId, includeMinimap: includeMinimap)
        return try await post("\(prefix)/render/\(jobId)", body: req)
    }

    // MARK: - Export

    public func export(jobId: String) async throws -> ExportResponse {
        return try await get("\(prefix)/export/\(jobId)")
    }

    // MARK: - Job Status

    public func jobStatus(jobId: String) async throws -> JobStatusResponse {
        return try await get("\(prefix)/jobs/status/\(jobId)")
    }

    public func jobsList() async throws -> [JobStatusResponse] {
        return try await get("\(prefix)/jobs/list")
    }

    // MARK: - Polling

    /// Poll a job until it reaches `completed` or `failed`.
    /// - Parameters:
    ///   - jobId: The job identifier to poll.
    ///   - interval: Seconds between polls (default 2).
    ///   - timeout: Max seconds to wait (default 300 = 5 min).
    /// - Returns: Final `JobStatusResponse` with status `completed`.
    /// - Throws: `AthLinkError.timeout` or `AthLinkError.jobFailed`.
    public func pollUntilDone(
        jobId: String,
        interval: TimeInterval = 2.0,
        timeout: TimeInterval = 300.0
    ) async throws -> JobStatusResponse {
        let deadline = Date().addingTimeInterval(timeout)

        while Date() < deadline {
            let status = try await jobStatus(jobId: jobId)

            switch status.status {
            case "completed":
                return status
            case "failed":
                throw AthLinkError.jobFailed(status.error ?? "Unknown error")
            default:
                try await Task.sleep(nanoseconds: UInt64(interval * 1_000_000_000))
            }
        }
        throw AthLinkError.timeout
    }

    // MARK: - Full Pipeline

    /// Run the full analysis pipeline: track → pitch → tactics.
    /// Polls each async step to completion before starting the next.
    /// - Returns: The aggregated `ExportResponse`.
    public func runFullPipeline(
        jobId: String,
        videoPath: String,
        frameStride: Int = 5,
        maxFrames: Int? = nil,
        pollInterval: TimeInterval = 2.0,
        stepTimeout: TimeInterval = 600.0
    ) async throws -> ExportResponse {
        // Step 1: Track + Teams
        let trackReq = TrackPlayersRequest(
            jobId: jobId, videoPath: videoPath,
            frameStride: frameStride, maxFrames: maxFrames
        )
        let trackJob = try await trackPlayersWithTeams(trackReq)
        _ = try await pollUntilDone(jobId: trackJob.jobId,
                                    interval: pollInterval, timeout: stepTimeout)

        // Step 2: Pitch mapping
        let pitchReq = PitchMapRequest(
            jobId: jobId, videoPath: videoPath, frameStride: frameStride
        )
        let pitchJob = try await pitchMap(pitchReq)
        _ = try await pollUntilDone(jobId: pitchJob.jobId,
                                    interval: pollInterval, timeout: stepTimeout)

        // Step 3: Tactics
        let tacticsReq = TacticsRequest(jobId: jobId)
        let tacticsJob = try await tacticsAnalyze(tacticsReq)
        _ = try await pollUntilDone(jobId: tacticsJob.jobId,
                                    interval: pollInterval, timeout: stepTimeout)

        // Step 4: Export
        return try await export(jobId: jobId)
    }

    // MARK: - Private Helpers

    private func get<T: Decodable>(_ url: String) async throws -> T {
        guard let requestURL = URL(string: url) else {
            throw AthLinkError.invalidURL(url)
        }
        let (data, response) = try await session.data(from: requestURL)
        try checkHTTPResponse(response, data: data)
        return try decode(data)
    }

    private func post<B: Encodable, T: Decodable>(
        _ url: String, body: B
    ) async throws -> T {
        guard let requestURL = URL(string: url) else {
            throw AthLinkError.invalidURL(url)
        }
        var request = URLRequest(url: requestURL)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        request.httpBody = try JSONEncoder().encode(body)

        let (data, response) = try await session.data(for: request)
        try checkHTTPResponse(response, data: data)
        return try decode(data)
    }

    private func checkHTTPResponse(_ response: URLResponse, data: Data) throws {
        guard let http = response as? HTTPURLResponse else { return }
        guard (200...299).contains(http.statusCode) else {
            let detail: String
            if let apiErr = try? JSONDecoder().decode(APIError.self, from: data) {
                detail = apiErr.detail
            } else {
                detail = String(data: data, encoding: .utf8) ?? "Unknown error"
            }
            throw AthLinkError.httpError(statusCode: http.statusCode, detail: detail)
        }
    }

    private func decode<T: Decodable>(_ data: Data) throws -> T {
        do {
            return try JSONDecoder().decode(T.self, from: data)
        } catch {
            throw AthLinkError.decodingFailed(error)
        }
    }
}
