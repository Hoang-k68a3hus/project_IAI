import axios from "axios";
import { VIECOMREC_BASEURL } from "./BaseURLs";

/**
 * VieComRec API Client
 * Tích hợp với hệ thống gợi ý mỹ phẩm AI
 */

const viecomrecAPI = axios.create({
  baseURL: VIECOMREC_BASEURL,
  timeout: 10000,
  headers: {
    "Content-Type": "application/json",
  },
});

/**
 * POST /recommend - Gợi ý sản phẩm cho user
 * @param {number} userId - ID của user
 * @param {number} topk - Số lượng recommendations (default: 10)
 * @param {boolean} excludeSeen - Loại bỏ sản phẩm đã mua (default: true)
 * @param {object} filterParams - Filters (brand, category, min_price, max_price)
 */
export const getRecommendations = async (
  userId,
  topk = 10,
  excludeSeen = true,
  filterParams = null
) => {
  try {
    const response = await viecomrecAPI.post("/recommend", {
      user_id: userId,
      topk,
      exclude_seen: excludeSeen,
      filter_params: filterParams,
      rerank: true,
    });
    return response.data;
  } catch (error) {
    console.error("VieComRec recommend error:", error);
    return null;
  }
};

/**
 * POST /batch_recommend - Gợi ý cho nhiều users
 * @param {number[]} userIds - Danh sách user IDs
 * @param {number} topk - Số lượng recommendations
 */
export const getBatchRecommendations = async (userIds, topk = 10) => {
  try {
    const response = await viecomrecAPI.post("/batch_recommend", {
      user_ids: userIds,
      topk,
      exclude_seen: true,
    });
    return response.data;
  } catch (error) {
    console.error("VieComRec batch recommend error:", error);
    return null;
  }
};

/**
 * POST /search - Tìm kiếm semantic tiếng Việt
 * @param {string} query - Query tìm kiếm
 * @param {number} topk - Số kết quả
 * @param {object} filters - Filters (brand, category, min_price, max_price)
 */
export const semanticSearch = async (query, topk = 10, filters = null) => {
  try {
    const response = await viecomrecAPI.post("/search", {
      query,
      topk,
      filters,
      rerank: true,
    });
    return response.data;
  } catch (error) {
    console.error("VieComRec search error:", error);
    return null;
  }
};

/**
 * POST /similar_items - Tìm sản phẩm tương tự (CF-based)
 * @param {number} productId - ID sản phẩm
 * @param {number} topk - Số lượng kết quả
 */
export const getSimilarItems = async (productId, topk = 10) => {
  try {
    const response = await viecomrecAPI.post("/similar_items", {
      product_id: productId,
      topk,
      use_cf: true,
    });
    return response.data;
  } catch (error) {
    console.error("VieComRec similar items error:", error);
    return null;
  }
};

/**
 * POST /search/similar - Tìm sản phẩm tương tự theo nội dung
 * @param {number} productId - ID sản phẩm
 * @param {number} topk - Số lượng kết quả
 */
export const getContentSimilar = async (productId, topk = 10) => {
  try {
    const response = await viecomrecAPI.post("/search/similar", {
      product_id: productId,
      topk,
      exclude_self: true,
    });
    return response.data;
  } catch (error) {
    console.error("VieComRec content similar error:", error);
    return null;
  }
};

/**
 * POST /search/profile - Tìm kiếm dựa trên lịch sử mua hàng
 * @param {number[]} productHistory - Danh sách product IDs đã mua
 * @param {number} topk - Số kết quả
 */
export const getProfileBasedSearch = async (productHistory, topk = 10, filters = null) => {
  try {
    const response = await viecomrecAPI.post("/search/profile", {
      product_history: productHistory,
      topk,
      exclude_history: true,
      filters,
    });
    return response.data;
  } catch (error) {
    console.error("VieComRec profile search error:", error);
    return null;
  }
};

/**
 * GET /search/filters - Lấy danh sách filters
 */
export const getSearchFilters = async () => {
  try {
    const response = await viecomrecAPI.get("/search/filters");
    return response.data;
  } catch (error) {
    console.error("VieComRec filters error:", error);
    return null;
  }
};

/**
 * GET /health - Kiểm tra trạng thái service
 */
export const checkHealth = async () => {
  try {
    const response = await viecomrecAPI.get("/health");
    return response.data;
  } catch (error) {
    console.error("VieComRec health check error:", error);
    return null;
  }
};

/**
 * GET /model_info - Thông tin chi tiết về model
 */
export const getModelInfo = async () => {
  try {
    const response = await viecomrecAPI.get("/model_info");
    return response.data;
  } catch (error) {
    console.error("VieComRec model info error:", error);
    return null;
  }
};

// ============== SCHEDULER API ==============

/**
 * GET /scheduler/status - Lấy trạng thái scheduler
 */
export const getSchedulerStatus = async () => {
  try {
    const response = await viecomrecAPI.get("/scheduler/status");
    return response.data;
  } catch (error) {
    console.error("VieComRec scheduler status error:", error);
    return null;
  }
};

/**
 * POST /scheduler/start - Khởi động scheduler
 */
export const startScheduler = async () => {
  try {
    const response = await viecomrecAPI.post("/scheduler/start");
    return response.data;
  } catch (error) {
    console.error("VieComRec scheduler start error:", error);
    return null;
  }
};

/**
 * POST /scheduler/stop - Dừng scheduler
 */
export const stopScheduler = async () => {
  try {
    const response = await viecomrecAPI.post("/scheduler/stop");
    return response.data;
  } catch (error) {
    console.error("VieComRec scheduler stop error:", error);
    return null;
  }
};

/**
 * POST /scheduler/trigger - Kích hoạt training thủ công
 * @param {string} modelType - Loại model: "phobert" hoặc "als"
 */
export const triggerTraining = async (modelType = "als") => {
  try {
    const response = await viecomrecAPI.post("/scheduler/trigger", null, {
      params: { model_type: modelType }
    });
    return response.data;
  } catch (error) {
    console.error("VieComRec trigger training error:", error);
    return null;
  }
};

/**
 * GET /scheduler/history - Lấy lịch sử training
 * @param {number} limit - Số lượng records (default: 10)
 */
export const getTrainingHistory = async (limit = 10) => {
  try {
    const response = await viecomrecAPI.get("/scheduler/history", {
      params: { limit }
    });
    return response.data;
  } catch (error) {
    console.error("VieComRec training history error:", error);
    return null;
  }
};

/**
 * GET /scheduler/next-run - Lấy thời gian chạy tiếp theo
 */
export const getNextRun = async () => {
  try {
    const response = await viecomrecAPI.get("/scheduler/next-run");
    return response.data;
  } catch (error) {
    console.error("VieComRec next run error:", error);
    return null;
  }
};

/**
 * PUT /scheduler/config - Cập nhật cấu hình scheduler
 * @param {object} config - { interval_hours, enabled, auto_retrain_on_drift }
 */
export const updateSchedulerConfig = async (config) => {
  try {
    const response = await viecomrecAPI.put("/scheduler/config", config);
    return response.data;
  } catch (error) {
    console.error("VieComRec update config error:", error);
    return null;
  }
};

// ============== DRIFT DETECTION API ==============

/**
 * GET /drift/status - Lấy trạng thái drift detection
 */
export const getDriftStatus = async () => {
  try {
    const response = await viecomrecAPI.get("/drift/status");
    return response.data;
  } catch (error) {
    console.error("VieComRec drift status error:", error);
    return null;
  }
};

/**
 * POST /drift/check - Kiểm tra drift manually
 */
export const checkDrift = async () => {
  try {
    const response = await viecomrecAPI.post("/drift/check");
    return response.data;
  } catch (error) {
    console.error("VieComRec check drift error:", error);
    return null;
  }
};

export default viecomrecAPI;
