import express from "express";
import axios from "axios";
import { VIECOMREC_BASEURL } from "../services/BaseURLs.js";

const router = express.Router();

/**
 * POST /api/recommend
 * Proxy đến VieComRec API /recommend
 */
router.post("/", async (req, res) => {
  const { user_id, topk = 10, exclude_seen = true, filter_params = null } = req.body;

  try {
    const response = await axios.post(`${VIECOMREC_BASEURL}/recommend`, {
      user_id,
      topk,
      exclude_seen,
      filter_params,
      rerank: true,
    });

    res.json(response.data);
  } catch (error) {
    console.error("VieComRec API error:", error.message);
    
    // Fallback to mock data if VieComRec is unavailable
    const mockRecommendations = [
      {
        rank: 1,
        product_id: 101,
        score: 0.91,
        product_name: "Serum Vitamin C",
        brand: "Balance",
        category: "serum",
        price: 120000,
        avg_rating: 4.8,
        num_sold: 3500,
      },
      {
        rank: 2,
        product_id: 203,
        score: 0.85,
        product_name: "Sữa rửa mặt HadaLabo",
        brand: "Hada Labo",
        category: "cleanser",
        price: 89000,
        avg_rating: 4.6,
        num_sold: 12000,
      },
      {
        rank: 3,
        product_id: 331,
        score: 0.82,
        product_name: "Kem dưỡng Simple",
        brand: "Simple",
        category: "moisturizer",
        price: 160000,
        avg_rating: 4.7,
        num_sold: 5000,
      },
    ];

    res.json({
      user_id: user_id,
      recommendations: mockRecommendations,
      count: mockRecommendations.length,
      is_fallback: true,
      fallback_method: "mock_data",
      latency_ms: 0,
      model_id: "fallback_mock",
    });
  }
});

/**
 * POST /api/recommend/search
 * Proxy đến VieComRec API /search (semantic search)
 */
router.post("/search", async (req, res) => {
  const { query, topk = 10, filters = null } = req.body;

  try {
    const response = await axios.post(`${VIECOMREC_BASEURL}/search`, {
      query,
      topk,
      filters,
      rerank: true,
    });

    res.json(response.data);
  } catch (error) {
    console.error("VieComRec search error:", error.message);
    res.status(500).json({ 
      error: "Search service unavailable",
      message: error.message 
    });
  }
});

/**
 * POST /api/recommend/similar
 * Proxy đến VieComRec API /similar_items
 */
router.post("/similar", async (req, res) => {
  const { product_id, topk = 10 } = req.body;

  try {
    const response = await axios.post(`${VIECOMREC_BASEURL}/similar_items`, {
      product_id,
      topk,
      use_cf: true,
    });

    res.json(response.data);
  } catch (error) {
    console.error("VieComRec similar items error:", error.message);
    res.status(500).json({ 
      error: "Similar items service unavailable",
      message: error.message 
    });
  }
});

/**
 * GET /api/recommend/health
 * Proxy đến VieComRec API /health
 */
router.get("/health", async (req, res) => {
  try {
    const response = await axios.get(`${VIECOMREC_BASEURL}/health`);
    res.json(response.data);
  } catch (error) {
    res.status(503).json({ 
      status: "unhealthy",
      error: "VieComRec service unavailable",
      message: error.message 
    });
  }
});

export default router;
