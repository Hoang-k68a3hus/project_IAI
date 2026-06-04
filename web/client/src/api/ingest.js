import axios from "axios";
import { INGEST_BASEURL } from "./BaseURLs";

/**
 * Ingest API Client - gửi purchase & review events đến ML
 */

/**
 * POST /ingest/purchase
 * Gửi thông tin mua hàng
 */
export const sendPurchaseEvent = async (userId, productId, quantity = 1, orderId = null) => {
  try {
    const response = await axios.post(`${INGEST_BASEURL}/purchase`, {
      user_id: userId,
      product_id: productId,
      quantity,
      order_id: orderId,
    });
    return response.data;
  } catch (error) {
    console.error("Purchase event error:", error);
    return { status: "error", message: error.message };
  }
};

/**
 * POST /ingest/review
 * Gửi review/đánh giá
 */
export const sendReviewEvent = async (userId, productId, rating, comment = "", orderId = null) => {
  try {
    const response = await axios.post(`${INGEST_BASEURL}/review`, {
      user_id: userId,
      product_id: productId,
      rating: parseFloat(rating),
      comment,
      order_id: orderId,
    });
    return response.data;
  } catch (error) {
    console.error("Review event error:", error);
    return { status: "error", message: error.message };
  }
};

/**
 * POST /ingest/batch
 * Batch send multiple events
 */
export const sendBatchEvents = async (reviews = [], purchases = []) => {
  try {
    const response = await axios.post(`${INGEST_BASEURL}/batch`, {
      reviews,
      purchases,
    });
    return response.data;
  } catch (error) {
    console.error("Batch ingest error:", error);
    return { status: "error", message: error.message };
  }
};

/**
 * GET /ingest/stats
 * Lấy thống kê ingest
 */
export const getIngestStats = async () => {
  try {
    const response = await axios.get(`${INGEST_BASEURL}/stats`);
    return response.data;
  } catch (error) {
    console.error("Ingest stats error:", error);
    return null;
  }
};

const ingestAPI = {
  sendPurchaseEvent,
  sendReviewEvent,
  sendBatchEvents,
  getIngestStats,
};

export default ingestAPI;
