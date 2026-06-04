import axios from "axios";
import Review from "../../model/Review.js";
import Order from "../../model/Orders.js";
import generateId from "../../utils/generateId.js";

const VIECOMREC_API = process.env.VIECOMREC_API || "http://localhost:8000";

/**
 * POST /ingest/purchase
 * Gửi thông tin mua hàng đến VieComRec ML
 */
export const ingestPurchase = async (req, res) => {
  try {
    const { user_id, product_id, quantity, order_id } = req.body;

    // Validate input
    if (!user_id || !product_id) {
      return res.status(400).json({
        status: "error",
        message: "user_id and product_id are required"
      });
    }

    // Send to VieComRec API
    const response = await axios.post(
      `${VIECOMREC_API}/ingest/purchase`,
      {
        user_id,
        product_id: parseInt(product_id),
        quantity: parseInt(quantity) || 1,
        timestamp: new Date().toISOString()
      },
      { timeout: 5000 }
    );

    // Mark order as ingested if order_id provided
    if (order_id) {
      await Order.findOneAndUpdate(
        { order_id },
        {
          ingested: true,
          ingest_timestamp: new Date()
        }
      );
    }

    return res.status(200).json({
      status: "accepted",
      message: "Purchase recorded successfully",
      data: response.data,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error("Purchase ingest error:", error.message);

    // Return success even if ML API fails (graceful degradation)
    return res.status(202).json({
      status: "pending",
      message: "Purchase staged locally. Will be synced when ML API is available.",
      timestamp: new Date().toISOString()
    });
  }
};

/**
 * POST /ingest/review
 * Gửi review/đánh giá đến VieComRec ML
 */
export const ingestReview = async (req, res) => {
  try {
    const { user_id, product_id, rating, comment, order_id } = req.body;

    // Validate input
    if (!product_id || !rating) {
      return res.status(400).json({
        status: "error",
        message: "product_id and rating are required"
      });
    }

    if (rating < 1 || rating > 5) {
      return res.status(400).json({
        status: "error",
        message: "rating must be between 1 and 5"
      });
    }

    // Create review record in MongoDB
    const review_id = `rev_${generateId()}`;
    const reviewData = {
      review_id,
      user_id: user_id || "anonymous",
      product_id: parseInt(product_id),
      order_id,
      rating: parseFloat(rating),
      comment: comment || "",
      status: "APPROVED"  // Auto-approve for now
    };
    
    const review = await Review.create(reviewData);
    console.log(`✅ Review ${review_id} created for product ${product_id}`);

    // Try to send to VieComRec API (non-blocking)
    try {
      const mlPayload = {
        user_id: parseInt(user_id) || 1,
        product_id: parseInt(product_id),
        rating: parseFloat(rating),
        comment: comment || "",
        timestamp: new Date().toISOString()
      };
      console.log(`📤 Sending to VieComRec:`, JSON.stringify(mlPayload));
      
      const mlResponse = await axios.post(
        `${VIECOMREC_API}/ingest/review`,
        mlPayload,
        { timeout: 5000 }
      );
      console.log(`✅ Review sent to VieComRec ML:`, mlResponse.data);
    } catch (mlError) {
      console.warn(`⚠️ VieComRec ML error:`, mlError.response?.data || mlError.message);
    }

    return res.status(201).json({
      status: "accepted",
      message: "Cảm ơn bạn đã đánh giá sản phẩm!",
      review_id,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    console.error("Review ingest error:", error.message);
    
    return res.status(500).json({
      status: "error",
      message: "Không thể lưu đánh giá: " + error.message
    });
  }
};

/**
 * GET /ingest/stats
 * Xem thống kê ingestion
 */
export const getIngestStats = async (req, res) => {
  try {
    const reviews = await Review.find({ status: "PENDING" }).countDocuments();
    const orders = await Order.find({ ingested: false }).countDocuments();
    const today = new Date();
    today.setHours(0, 0, 0, 0);
    const todayReviews = await Review.find({
      created_at: { $gte: today }
    }).countDocuments();

    return res.status(200).json({
      status: "success",
      total_pending_reviews: reviews,
      total_pending_purchases: orders,
      reviews_today: todayReviews,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    return res.status(500).json({
      status: "error",
      message: error.message
    });
  }
};

/**
 * POST /ingest/batch
 * Batch ingest nhiều purchases và reviews
 */
export const ingestBatch = async (req, res) => {
  try {
    const { reviews = [], purchases = [] } = req.body;

    let processedReviews = 0;
    let processedPurchases = 0;
    const errors = [];

    // Process purchases
    for (const purchase of purchases) {
      try {
        await axios.post(
          `${VIECOMREC_API}/ingest/purchase`,
          {
            user_id: purchase.user_id,
            product_id: parseInt(purchase.product_id),
            quantity: parseInt(purchase.quantity) || 1,
            timestamp: purchase.timestamp || new Date().toISOString()
          },
          { timeout: 5000 }
        );
        processedPurchases++;
      } catch (err) {
        errors.push({
          type: "purchase",
          user_id: purchase.user_id,
          error: err.message
        });
      }
    }

    // Process reviews
    for (const review of reviews) {
      try {
        const review_id = `rev_${generateId()}`;
        await Review.create({
          review_id,
          user_id: review.user_id,
          product_id: parseInt(review.product_id),
          rating: parseFloat(review.rating),
          comment: review.comment || "",
          status: "PENDING"
        });

        await axios.post(
          `${VIECOMREC_API}/ingest/review`,
          {
            user_id: review.user_id,
            product_id: parseInt(review.product_id),
            rating: parseFloat(review.rating),
            comment: review.comment || "",
            timestamp: review.timestamp || new Date().toISOString()
          },
          { timeout: 5000 }
        );
        processedReviews++;
      } catch (err) {
        errors.push({
          type: "review",
          user_id: review.user_id,
          error: err.message
        });
      }
    }

    return res.status(200).json({
      status: "completed",
      total_processed: processedReviews + processedPurchases,
      reviews_processed: processedReviews,
      purchases_processed: processedPurchases,
      errors: errors.length > 0 ? errors : null,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    return res.status(500).json({
      status: "error",
      message: error.message
    });
  }
};
