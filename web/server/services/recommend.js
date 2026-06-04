import axios from "axios";
import { SERVER_BASE_URL } from "./BaseURLs.js";

export const getRecommendations = async (user_id, cart = []) => {
  try {
    const res = await axios.post(`${SERVER_BASE_URL}/api/recommend`, {
      user_id,
      cart,
    });

    return res.data;
  } catch (err) {
    console.error("Recommendation API error:", err);
    return null;
  }
};
