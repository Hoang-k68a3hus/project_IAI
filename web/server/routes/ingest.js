import express from "express";
import auth from "../middleware/auth.js";
import {
  ingestPurchase,
  ingestReview,
  getIngestStats,
  ingestBatch,
} from "../controller/me/Ingest.js";

const router = express.Router();

// Public endpoints (from payment gateway or ML API)
router.post("/purchase", ingestPurchase);
router.post("/review", ingestReview);
router.post("/batch", ingestBatch);

// Protected endpoints (admin only)
router.get("/stats", auth, getIngestStats);

export default router;
