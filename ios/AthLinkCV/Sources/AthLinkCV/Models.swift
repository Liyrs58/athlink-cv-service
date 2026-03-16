import Foundation

// MARK: - Health

public struct HealthResponse: Codable, Sendable {
    public let status: String
    public let service: String
    public let device: String
}

// MARK: - Job Queue

public struct QueuedJobResponse: Codable, Sendable {
    public let jobId: String
    public let status: String
    public let message: String
}

public struct JobStatusResponse: Codable, Sendable {
    public let jobId: String
    public let status: String          // queued | processing | completed | failed
    public let createdAt: Double
    public let startedAt: Double?
    public let completedAt: Double?
    public let error: String?
    // result is dynamic JSON — decode separately if needed
}

// MARK: - Track Request / Response

public struct TrackPlayersRequest: Codable, Sendable {
    public let jobId: String
    public let videoPath: String
    public var frameStride: Int = 5
    public var maxFrames: Int?
    public var maxTrackAge: Int = 10

    public init(jobId: String, videoPath: String,
                frameStride: Int = 5, maxFrames: Int? = nil,
                maxTrackAge: Int = 10) {
        self.jobId = jobId
        self.videoPath = videoPath
        self.frameStride = frameStride
        self.maxFrames = maxFrames
        self.maxTrackAge = maxTrackAge
    }
}

public struct TrackPlayersResponse: Codable, Sendable {
    public let jobId: String
    public let videoPath: String
    public let frameStride: Int
    public let framesProcessed: Int
    public let trackCount: Int
    public let outputPath: String
}

// MARK: - Pitch Mapping

public struct PitchMapRequest: Codable, Sendable {
    public let jobId: String
    public let videoPath: String
    public var frameStride: Int = 5
    public var maxFrames: Int?

    public init(jobId: String, videoPath: String,
                frameStride: Int = 5, maxFrames: Int? = nil) {
        self.jobId = jobId
        self.videoPath = videoPath
        self.frameStride = frameStride
        self.maxFrames = maxFrames
    }
}

// MARK: - Tactics

public struct TacticsRequest: Codable, Sendable {
    public let jobId: String

    public init(jobId: String) {
        self.jobId = jobId
    }
}

// MARK: - Render

public struct RenderRequest: Codable, Sendable {
    public let jobId: String
    public var includeMinimap: Bool = false

    public init(jobId: String, includeMinimap: Bool = false) {
        self.jobId = jobId
        self.includeMinimap = includeMinimap
    }
}

// MARK: - Export (aggregated analysis result)

public struct ExportResponse: Codable, Sendable {
    public let jobId: String
    public let videoMeta: VideoMeta
    public let teams: TeamsSummary
    public let frames: [FrameData]
    public let tactics: TacticsData
    public let spaceOccupation: SpaceOccupation?
    public let events: [GameEvent]?
}

public struct VideoMeta: Codable, Sendable {
    public let width: Int
    public let height: Int
    public let fps: Double
    public let durationSeconds: Double
    public let frameCount: Int
}

public struct TeamsSummary: Codable, Sendable {
    public let team0: TeamInfo
    public let team1: TeamInfo
}

public struct TeamInfo: Codable, Sendable {
    public let color: String        // hex e.g. "#0064FF"
    public let playerCount: Int
}

public struct FrameData: Codable, Sendable {
    public let frameIndex: Int
    public let timestampSeconds: Double
    public let players: [PlayerFrame]
}

public struct PlayerFrame: Codable, Sendable {
    public let trackId: Int
    public let teamId: Int
    public let bbox: [Int]          // [x, y, w, h]
    public let pitchX: Double?
    public let pitchY: Double?
}

public struct TacticsData: Codable, Sendable {
    public let team0Formation: String?
    public let team1Formation: String?
    public let team0Heatmap: [[Double]]?
    public let team1Heatmap: [[Double]]?
}

public struct SpaceOccupation: Codable, Sendable {
    public let gridRows: Int
    public let gridCols: Int
    public let team0Cells: Int
    public let team1Cells: Int
    public let grid: [[Int]]
}

public struct GameEvent: Codable, Sendable {
    public let type: String         // "pass", "shot", "tackle"
    public let frameIndex: Int
    public let trackId: Int
    public let details: EventDetails?
}

public struct EventDetails: Codable, Sendable {
    public let fromX: Double?
    public let fromY: Double?
    public let toX: Double?
    public let toY: Double?
    public let distance: Double?
    public let targetTrackId: Int?
}

// MARK: - Error

public struct APIError: Codable, Sendable {
    public let detail: String
}
