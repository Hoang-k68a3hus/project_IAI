/**
 * VieComRec Recommendation API
 * File này đã được thay thế bởi viecomrec.js
 * Giữ lại để backward compatibility
 */

import { getRecommendations } from "./viecomrec";

export { getRecommendations };

// Legacy function - redirect to new API
export const loadRecommendations = async (userId = 1, cart = []) => {
  try {
    const result = await getRecommendations(userId, 10, true);
    
    if (result && result.recommendations) {
      return {
        recommendations: result.recommendations.map((item) => ({
          product_id: item.product_id,
          name: item.product_name,
          price: item.price,
          brand: item.brand,
          image: `/images/${item.product_id}.jpg`,
        })),
      };
    }
    return { recommendations: [] };
  } catch (err) {
    console.error("Recommendation API error:", err);
    return null;
  }
};
