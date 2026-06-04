import React, { useState } from "react";
import styles from "./ReviewForm.module.css";
import { sendReviewEvent } from "../../api/ingest";

const ReviewForm = ({ productId, userId, orderId, onSuccess, onClose }) => {
  const [rating, setRating] = useState(0);
  const [hoverRating, setHoverRating] = useState(0);
  const [comment, setComment] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [success, setSuccess] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError("");

    if (rating === 0) {
      setError("Vui lòng chọn đánh giá sao");
      return;
    }

    if (!comment.trim()) {
      setError("Vui lòng viết nhận xét");
      return;
    }

    setLoading(true);

    try {
      const result = await sendReviewEvent(
        userId,
        productId,
        rating,
        comment,
        orderId
      );

      if (result.status === "accepted" || result.status === "pending") {
        setSuccess(true);
        setTimeout(() => {
          onSuccess?.();
          onClose?.();
        }, 1500);
      } else {
        setError(result.message || "Không thể gửi review");
      }
    } catch (err) {
      setError("Lỗi khi gửi review: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  if (success) {
    return (
      <div className={styles.successMessage}>
        <div className={styles.successIcon}>✅</div>
        <p>Cảm ơn bạn đã đánh giá!</p>
        <small>Review của bạn sẽ được duyệt trong thời gian sớm nhất</small>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      <div className={styles.header}>
        <h3>Đánh giá sản phẩm</h3>
        <button
          className={styles.closeBtn}
          onClick={onClose}
          disabled={loading}
        >
          ✕
        </button>
      </div>

      <form onSubmit={handleSubmit} className={styles.form}>
        {/* Star Rating */}
        <div className={styles.ratingGroup}>
          <label>Chất lượng sản phẩm</label>
          <div className={styles.starContainer}>
            {[1, 2, 3, 4, 5].map((star) => (
              <button
                key={star}
                type="button"
                className={`${styles.star} ${
                  (hoverRating || rating) >= star ? styles.active : ""
                }`}
                onClick={() => setRating(star)}
                onMouseEnter={() => setHoverRating(star)}
                onMouseLeave={() => setHoverRating(0)}
              >
                ★
              </button>
            ))}
          </div>
          {rating > 0 && (
            <small className={styles.ratingLabel}>
              {rating === 5
                ? "Tuyệt vời!"
                : rating === 4
                ? "Rất tốt"
                : rating === 3
                ? "Bình thường"
                : rating === 2
                ? "Tệ"
                : "Rất tệ"}
            </small>
          )}
        </div>

        {/* Comment */}
        <div className={styles.commentGroup}>
          <label>Nhận xét chi tiết</label>
          <textarea
            value={comment}
            onChange={(e) => setComment(e.target.value)}
            placeholder="Chia sẻ trải nghiệm của bạn về sản phẩm này..."
            maxLength={500}
            rows={4}
            disabled={loading}
          />
          <small>
            {comment.length}/500 ký tự
          </small>
        </div>

        {/* Error Message */}
        {error && <div className={styles.error}>{error}</div>}

        {/* Buttons */}
        <div className={styles.buttonGroup}>
          <button
            type="button"
            className={styles.cancelBtn}
            onClick={onClose}
            disabled={loading}
          >
            Hủy
          </button>
          <button
            type="submit"
            className={styles.submitBtn}
            disabled={loading || rating === 0 || !comment.trim()}
          >
            {loading ? "⏳ Đang gửi..." : "✓ Gửi Review"}
          </button>
        </div>
      </form>
    </div>
  );
};

export default ReviewForm;
