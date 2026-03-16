import Foundation

// MARK: - Reports Extension

public extension AthLinkClient {
    
    /// Download raw PDF data for a player report.
    /// - Parameters:
    ///   - jobId: The job identifier
    ///   - trackId: The player's track ID
    /// - Returns: Raw PDF data
    /// - Throws: AthLinkError if download fails or status != 200
    func downloadPlayerReport(jobId: String, trackId: Int) async throws -> Data {
        let url = "\(prefix)/reports/\(jobId)/player/\(trackId)"
        guard let requestURL = URL(string: url) else {
            throw AthLinkError.invalidURL(url)
        }
        
        var request = URLRequest(url: requestURL)
        request.httpMethod = "GET"
        
        let (data, response) = try await session.data(for: request)
        
        guard let http = response as? HTTPURLResponse else {
            throw AthLinkError.httpError(statusCode: 0, detail: "Invalid response")
        }
        
        guard http.statusCode == 200 else {
            let detail: String
            if let apiErr = try? JSONDecoder().decode(APIError.self, from: data) {
                detail = apiErr.detail
            } else {
                detail = String(data: data, encoding: .utf8) ?? "Unknown error"
            }
            throw AthLinkError.httpError(statusCode: http.statusCode, detail: detail)
        }
        
        return data
    }
    
    /// Download and save a player report to disk.
    /// - Parameters:
    ///   - jobId: The job identifier
    ///   - trackId: The player's track ID
    ///   - url: Local URL where the PDF should be saved
    /// - Throws: AthLinkError if download or save fails
    func savePlayerReport(jobId: String, trackId: Int, to url: URL) async throws {
        let data = try await downloadPlayerReport(jobId: jobId, trackId: trackId)
        try data.write(to: url)
    }
    
    /// Get the remote URL for a player report (does not fetch the data).
    /// - Parameters:
    ///   - jobId: The job identifier
    ///   - trackId: The player's track ID
    /// - Returns: Remote URL for the player report
    func playerReportURL(jobId: String, trackId: Int) -> URL {
        return URL(string: "\(prefix)/reports/\(jobId)/player/\(trackId)")!
    }
    
    /// Download raw PDF data for a team report.
    /// - Parameters:
    ///   - jobId: The job identifier
    ///   - team: The team ID (0 or 1)
    /// - Returns: Raw PDF data
    /// - Throws: AthLinkError if download fails or status != 200
    func downloadTeamReport(jobId: String, team: Int) async throws -> Data {
        let url = "\(prefix)/reports/\(jobId)/team/\(team)"
        guard let requestURL = URL(string: url) else {
            throw AthLinkError.invalidURL(url)
        }
        
        var request = URLRequest(url: requestURL)
        request.httpMethod = "GET"
        
        let (data, response) = try await session.data(for: request)
        
        guard let http = response as? HTTPURLResponse else {
            throw AthLinkError.httpError(statusCode: 0, detail: "Invalid response")
        }
        
        guard http.statusCode == 200 else {
            let detail: String
            if let apiErr = try? JSONDecoder().decode(APIError.self, from: data) {
                detail = apiErr.detail
            } else {
                detail = String(data: data, encoding: .utf8) ?? "Unknown error"
            }
            throw AthLinkError.httpError(statusCode: http.statusCode, detail: detail)
        }
        
        return data
    }
    
    /// Download and save a team report to disk.
    /// - Parameters:
    ///   - jobId: The job identifier
    ///   - team: The team ID (0 or 1)
    ///   - url: Local URL where the PDF should be saved
    /// - Throws: AthLinkError if download or save fails
    func saveTeamReport(jobId: String, team: Int, to url: URL) async throws {
        let data = try await downloadTeamReport(jobId: jobId, team: team)
        try data.write(to: url)
    }
    
    /// Get the remote URL for a team report (does not fetch the data).
    /// - Parameters:
    ///   - jobId: The job identifier
    ///   - team: The team ID (0 or 1)
    /// - Returns: Remote URL for the team report
    func teamReportURL(jobId: String, team: Int) -> URL {
        return URL(string: "\(prefix)/reports/\(jobId)/team/\(team)")!
    }
}
