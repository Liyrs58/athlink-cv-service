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

    internal let baseURL: String
    internal let session: URLSession
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

    // MARK: - Full Match Pipeline

    public func runFullMatch(jobId: String, videoPath: String,
                           frameStride: Int, forceRestart: Bool) async throws -> QueuedJobResponse {
        let req = FullMatchRequest(jobId: jobId, videoPath: videoPath,
                                 frameStride: frameStride, forceRestart: forceRestart)
        return try await post("\(prefix)/match/run", body: req)
    }

    public func getMatchProgress(jobId: String) async throws -> MatchProgress {
        return try await get("\(prefix)/match/progress/\(jobId)")
    }

    public func uploadJobResults(jobId: String) async throws -> UploadResult {
        return try await post("\(prefix)/storage/upload/\(jobId)", body: EmptyRequest())
    }

    public func getStorageURL(jobId: String, bucket: String,
                             path: String) async throws -> StorageURL {
        let req = StorageURLRequest(bucket: bucket, path: path)
        return try await post("\(prefix)/storage/url/\(jobId)", body: req)
    }

    // MARK: - Analytics

    public func getPassNetwork(jobId: String) async throws -> PassNetwork {
        return try await get("\(prefix)/pass-network/\(jobId)")
    }

    public func getXG(jobId: String) async throws -> XGResult {
        return try await get("\(prefix)/xg/\(jobId)")
    }

    public func getHeatmap(jobId: String,
                          trackId: Int?) async throws -> HeatmapResult {
        let url: String
        if let trackId = trackId {
            url = "\(prefix)/heatmap/\(jobId)?trackId=\(trackId)"
        } else {
            url = "\(prefix)/heatmap/\(jobId)"
        }
        return try await get(url)
    }

    public func getPressing(jobId: String) async throws -> PressingResult {
        return try await get("\(prefix)/pressing/\(jobId)")
    }

    public func getFormation(jobId: String) async throws -> FormationResult {
        return try await get("\(prefix)/formation/\(jobId)")
    }

    // MARK: - Phase 3 Intelligence

    public func getEvents(jobId: String, type: EventType?,
                         team: Int?, fromSeconds: Double?,
                         toSeconds: Double?) async throws -> EventResult {
        var components = URLComponents(string: "\(prefix)/events/\(jobId)")!
        var queryItems: [URLQueryItem] = []
        
        if let type = type {
            queryItems.append(URLQueryItem(name: "type", value: type.rawValue))
        }
        if let team = team {
            queryItems.append(URLQueryItem(name: "team", value: "\(team)"))
        }
        if let fromSeconds = fromSeconds {
            queryItems.append(URLQueryItem(name: "from", value: "\(fromSeconds)"))
        }
        if let toSeconds = toSeconds {
            queryItems.append(URLQueryItem(name: "to", value: "\(toSeconds)"))
        }
        
        if !queryItems.isEmpty {
            components.queryItems = queryItems
        }
        
        return try await get(components.url?.absoluteString ?? "\(prefix)/events/\(jobId)")
    }

    public func getEventSummary(jobId: String) async throws -> EventSummary {
        return try await get("\(prefix)/events/\(jobId)/summary")
    }

    public func getDefensiveLine(jobId: String,
                                team: Int?,
                                summaryOnly: Bool) async throws -> DefensiveLineResult {
        var components = URLComponents(string: "\(prefix)/defensive-line/\(jobId)")!
        var queryItems: [URLQueryItem] = []
        
        if let team = team {
            queryItems.append(URLQueryItem(name: "team", value: "\(team)"))
        }
        queryItems.append(URLQueryItem(name: "summary", value: "\(summaryOnly)"))
        
        components.queryItems = queryItems
        
        return try await get(components.url?.absoluteString ?? "\(prefix)/defensive-line/\(jobId)")
    }

    public func getCounterPress(jobId: String,
                               team: Int?,
                               intensity: PressIntensity?,
                               outcome: PressOutcome?) async throws -> CounterPressResult {
        var components = URLComponents(string: "\(prefix)/counter-press/\(jobId)")!
        var queryItems: [URLQueryItem] = []
        
        if let team = team {
            queryItems.append(URLQueryItem(name: "team", value: "\(team)"))
        }
        if let intensity = intensity {
            queryItems.append(URLQueryItem(name: "intensity", value: intensity.rawValue))
        }
        if let outcome = outcome {
            queryItems.append(URLQueryItem(name: "outcome", value: outcome.rawValue))
        }
        
        components.queryItems = queryItems
        
        return try await get(components.url?.absoluteString ?? "\(prefix)/counter-press/\(jobId)")
    }

    public func getSetPieces(jobId: String,
                            type: SetPieceType?,
                            team: Int?) async throws -> SetPieceResult {
        var components = URLComponents(string: "\(prefix)/set-pieces/\(jobId)")!
        var queryItems: [URLQueryItem] = []
        
        if let type = type {
            queryItems.append(URLQueryItem(name: "type", value: type.rawValue))
        }
        if let team = team {
            queryItems.append(URLQueryItem(name: "team", value: "\(team)"))
        }
        
        components.queryItems = queryItems
        
        return try await get(components.url?.absoluteString ?? "\(prefix)/set-pieces/\(jobId)")
    }

    public func getAnalytics(jobId: String) async throws -> AnalyticsReport {
        return try await get("\(prefix)/analytics/\(jobId)")
    }

    public func getAvailableServices(jobId: String) async throws -> AvailableServices {
        return try await get("\(prefix)/analytics/\(jobId)/available")
    }

    public func getAvailableReports(jobId: String) async throws -> AvailableReports {
        return try await get("\(prefix)/reports/\(jobId)/available")
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
