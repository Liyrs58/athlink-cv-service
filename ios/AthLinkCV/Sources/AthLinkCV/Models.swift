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

// MARK: - Additional Request Models

public struct FullMatchRequest: Codable, Sendable {
    public let jobId: String
    public let videoPath: String
    public let frameStride: Int
    public let forceRestart: Bool
    
    public init(jobId: String, videoPath: String,
                frameStride: Int, forceRestart: Bool) {
        self.jobId = jobId
        self.videoPath = videoPath
        self.frameStride = frameStride
        self.forceRestart = forceRestart
    }
}

public struct EmptyRequest: Codable, Sendable {
    public init() {}
}

public struct StorageURLRequest: Codable, Sendable {
    public let bucket: String
    public let path: String
    
    public init(bucket: String, path: String) {
        self.bucket = bucket
        self.path = path
    }
}

// MARK: - Analytics Enums

public enum EventType: String, CaseIterable, Codable, Sendable {
    case pass = "pass"
    case shot = "shot"
    case dribble = "dribble"
    case turnover = "turnover"
    case clearance = "clearance"
    case ballRecovery = "ballRecovery"
    case carry = "carry"
}

public enum PressIntensity: String, CaseIterable, Codable, Sendable {
    case high = "high"
    case medium = "medium"
    case low = "low"
    case noPress = "noPress"
}

public enum PressOutcome: String, CaseIterable, Codable, Sendable {
    case successRegain = "successRegain"
    case failPossessionLost = "failPossessionLost"
    case failShotConceded = "failShotConceded"
}

public enum SetPieceType: String, CaseIterable, Codable, Sendable {
    case corner = "corner"
    case freeKick = "freeKick"
    case throwIn = "throwIn"
    case penalty = "penalty"
    case goalKick = "goalKick"
    case kickOff = "kickOff"
}

// MARK: - Full Match Pipeline

public struct MatchProgress: Codable, Sendable {
    public let job_id: String
    public let status: String
    public let progress_pct: Double
    public let last_completed_second: Double
    public let total_seconds_estimated: Double
    public let chunks_done: Int
    public let resumed: Bool
}

public struct UploadResult: Codable, Sendable {
    public let success: Bool
    public let uploadedFiles: [String]?
    public let error: String?
}

public struct StorageURL: Codable, Sendable {
    public let url: String
    public let expiresAt: Double?
}

// MARK: - Phase 1 Analytics

public struct PassNetwork: Codable, Sendable {
    public let nodes: [PassNode]
    public let edges: [PassEdge]
    public let totalPasses: Int
    public let contestedFrames: Int
    public let passesByTeam: [String: Int]
    public let warning: String?
}

public struct PassNode: Codable, Sendable {
    public let id: String
    public let trackId: Int
    public let team: Int
    public let position: [Double]?
    public let passCount: Int?
}

public struct PassEdge: Codable, Sendable {
    public let source: String
    public let target: String
    public let count: Int
    public let avgDistance: Double?
}

public struct XGResult: Codable, Sendable {
    public let shots: [XGShot]
    public let xg_team_0: Double
    public let xg_team_1: Double
    public let shots_team_0: Int
    public let shots_team_1: Int
    public let ball_lost_frames: Int
    public let interpolated_frames: Int
}

public struct XGShot: Codable, Sendable {
    public let frame: Int
    public let ball_x: Double
    public let ball_y: Double
    public let dist_to_goal_m: Double
    public let angle_rad: Double
    public let xg: Double
    public let team: Int
    public let ball_speed_ms: Double
    public let interpolated_position: Bool
}

public struct HeatmapResult: Codable, Sendable {
    public let players: [String: PlayerHeatmap]
    public let team: [String: TeamHeatmapSummary]
}

public struct PlayerHeatmap: Codable, Sendable {
    public let team: Int
    public let total_distance_m: Double
    public let sprint_distance_m: Double
    public let high_intensity_distance_m: Double
    public let top_speed_ms: Double
    public let avg_speed_ms: Double
    public let sprint_count: Int
    public let position_heatmap: [[Double]]
    public let sprint_heatmap: [[Double]]
    public let high_intensity_heatmap: [[Double]]
    public let heatmap_grid_w: Int
    public let heatmap_grid_h: Int
}

public struct TeamHeatmapSummary: Codable, Sendable {
    public let total_distance_m: Double
    public let sprint_distance_m: Double
    public let top_speed_ms: Double
    public let avg_distance_per_player_m: Double
}

public struct PressingResult: Codable, Sendable {
    public let team_0: TeamPressingData
    public let team_1: TeamPressingData
    public let frame_count_analysed: Int
    public let contested_frames: Int
}

public struct TeamPressingData: Codable, Sendable {
    public let ppda: Double?
    public let ppda_sample_size: Int
    public let defensive_actions: Int
    public let defensive_actions_in_def_third: Int
    public let passes_conceded_in_def_third: Int
    public let pressing_intensity_map: [Double]
    public let avg_press_height_m: Double
    public let avg_recovery_time_frames: Double
}

public struct FormationResult: Codable, Sendable {
    public let team_0: TeamFormation
    public let team_1: TeamFormation
    public let shape_shift_events: [ShapeShiftEvent]
}

public struct TeamFormation: Codable, Sendable {
    public let dominant_formation: String
    public let formation_stability: Double
    public let avg_compactness_m: Double
    public let timeline: [FormationWindow]
}

public struct FormationWindow: Codable, Sendable {
    public let frame_start: Int
    public let frame_end: Int
    public let formation: String
    public let raw_d_m_a: [Int]
    public let stability_score: Double
    public let def_centroid_x: Double
    public let mid_centroid_x: Double
    public let att_centroid_x: Double
    public let compactness_m: Double
}

public struct ShapeShiftEvent: Codable, Sendable {
    public let frame: Int
    public let team: Int
    public let from_formation: String
    public let to_formation: String
    public let type: String
}

// MARK: - Phase 3 Intelligence

public struct EventResult: Codable, Sendable {
    public let events: [MatchEvent]
    public let warning: String?
}

public struct MatchEvent: Codable, Sendable {
    public let frame: Int
    public let time: Double
    public let type: String
    public let team: Int
    public let player: Int
    public let position: [Double]?
    public let details: EventDetails?
    public let success: Bool?
}

public struct EventSummary: Codable, Sendable {
    public let team_0: TeamEventSummary
    public let team_1: TeamEventSummary
    public let total_events: Int
    public let duration_seconds: Double
}

public struct TeamEventSummary: Codable, Sendable {
    public let pass_count: Int
    public let shot_count: Int
    public let dribble_count: Int
    public let dribble_success_rate: Double
    public let turnover_count: Int
    public let clearance_count: Int
    public let forward_carry_distance_m: Double
}

public struct DefensiveLineResult: Codable, Sendable {
    public let team_0: TeamDefensiveLine
    public let team_1: TeamDefensiveLine
}

public struct TeamDefensiveLine: Codable, Sendable {
    public let avg_defensive_line_depth_m: Double
    public let avg_team_width_m: Double
    public let avg_team_length_m: Double
    public let avg_shape_area_m2: Double
    public let high_line_pct: Double
    public let low_block_pct: Double
    public let out_of_shape_events: [DefensiveLineFrame]
}

public struct DefensiveLineFrame: Codable, Sendable {
    public let frame_start: Int
    public let frame_end: Int
    public let reason: String
}

public struct CounterPressResult: Codable, Sendable {
    public let team_0: TeamCounterPress
    public let team_1: TeamCounterPress
    public let total_turnovers_analysed: Int
}

public struct TeamCounterPress: Codable, Sendable {
    public let total_attempts: Int
    public let high_intensity_pct: Double
    public let success_rate: Double
    public let avg_pressers: Double
    public let avg_time_to_first_press_s: Double
    public let intensity_map: [Double]
    public let windows: [CounterPressWindow]
}

public struct CounterPressWindow: Codable, Sendable {
    public let frame: Int
    public let intensity: String
    public let outcome: String?
    public let pressers: [Int]
    public let time_to_first_press_s: Double?
}

public struct SetPieceResult: Codable, Sendable {
    public let set_pieces: [SetPiece]
    public let summary: SetPieceSummary
    public let total_set_pieces: Int
}

public struct SetPiece: Codable, Sendable {
    public let frame: Int
    public let time: Double
    public let type: String
    public let team: Int
    public let player: Int?
    public let position: [Double]?
    public let outcome: String?
    public let details: SetPieceDetails?
}

public struct SetPieceDetails: Codable, Sendable {
    public let success: Bool?
    public let shot_created: Bool?
    public let pass_success: Bool?
}

public struct SetPieceSummary: Codable, Sendable {
    public let team_0: SetPieceTeamSummary
    public let team_1: SetPieceTeamSummary
}

public struct SetPieceTeamSummary: Codable, Sendable {
    public let corners: Int
    public let free_kicks: Int
    public let throw_ins: Int
    public let penalties: Int
    public let corner_shots_created: Int
    public let free_kick_shots_created: Int
}

// MARK: - Analytics & Reports

public struct AnalyticsReport: Codable, Sendable {
    public let match_summary: MatchSummary
    public let available_services: [String]
    public let errors: [String: String]?
}

public struct MatchSummary: Codable, Sendable {
    public let total_passes: Int?
    public let xg_team_0: Double?
    public let xg_team_1: Double?
    public let shots_team_0: Int?
    public let shots_team_1: Int?
    public let total_shots: Int?
    public let dribble_success_rate_team_0: Double?
    public let dribble_success_rate_team_1: Double?
    public let avg_def_line_depth_team_0: Double?
    public let avg_def_line_depth_team_1: Double?
    public let counter_press_success_rate_team_0: Double?
    public let counter_press_success_rate_team_1: Double?
    public let corners_team_0: Int?
    public let corners_team_1: Int?
    public let free_kicks_team_0: Int?
    public let free_kicks_team_1: Int?
    public let dominant_formation_team_0: String?
    public let dominant_formation_team_1: String?
    public let top_distance_player: TopPlayer?
    public let top_speed_player: TopPlayer?
}

public struct TopPlayer: Codable, Sendable {
    public let track_id: Int
    public let team: Int
    public let distance_m: Double?
    public let speed_ms: Double?
}

public struct AvailableServices: Codable, Sendable {
    public let services: [String]
}

public struct AvailableReports: Codable, Sendable {
    public let player_reports: [PlayerReportInfo]
    public let team_reports: [TeamReportInfo]
}

public struct PlayerReportInfo: Codable, Sendable {
    public let trackId: Int
    public let team: Int
    public let available: Bool
}

public struct TeamReportInfo: Codable, Sendable {
    public let team: Int
    public let available: Bool
}
