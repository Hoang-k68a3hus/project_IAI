import styles from "./orderId.module.css";
import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { useDispatch, useSelector } from "react-redux";
import Loading from "../../../components/loading/Loading";
import { fetchOrder } from "../../../actions/orders";
import ReviewForm from "../../../components/review-form/ReviewForm";
import { TRACKING_IMAGES } from "../../../shared/placeholders";

const OrderId = () => {
  const { id } = useParams();
  const order = useSelector((state) => state.orders.fetched);
  const user = useSelector((state) => state.authentication.user);
  const dispatch = useDispatch();
  const [loading, setLoading] = useState(true);
  const [showReviewModal, setShowReviewModal] = useState(false);
  const [selectedProductId, setSelectedProductId] = useState(null);
  const navigate = useNavigate();

  useEffect(() => {
    const onSuccess = () => {
      setLoading(false);
    };

    const onError = (e) => {
      setLoading(false);
      navigate("/404");
    };

    if (order && order.order_id === id) onSuccess();
    else dispatch(fetchOrder(id, onSuccess, onError));
  }, [dispatch, id, navigate, order]);

  const capitalizeFirst = (m) => {
    return m.charAt(0).toUpperCase() + m.slice(1).toLowerCase();
  };

  const getProgress = () => {
    switch (order?.status) {
      case "PROCESSING":
        return 50;
      case "FULFILLED":
        return 100;
      default:
        return 5;
    }
  };

  const openReviewModal = (productId) => {
    setSelectedProductId(productId);
    setShowReviewModal(true);
  };

  if (loading) return <Loading />;

  return (
    <div className={styles["wrapper"]}>
      <div className={"heading"}>
        <h1>Track Order</h1>
      </div>
      <div className={styles["sub"]}>
        Order <span>#{order.order_id}</span>
      </div>
      <div className={styles["full-progress"]}>
        <div
          className={styles["progress"]}
          style={{ width: getProgress() + "%" }}
        >
          <img
            className={styles["img"]}
            style={{
              transform: order.status === "CANCELLED" ? "scaleX(-1)" : "",
            }}
            src={TRACKING_IMAGES.order}
            alt={"Order"}
          />
        </div>
      </div>
      <div className={styles["status"]}>
        <span className={styles["update"]}>Order Update:</span> Your order has
        been {capitalizeFirst(order.status)}.
      </div>

      {/* Order Items */}
      {order?.status === "FULFILLED" && order?.products && (
        <div className={styles["products"]}>
          <h3>Sản phẩm đã mua</h3>
          <div className={styles["productsList"]}>
            {order.products.map((product, idx) => (
              <div key={idx} className={styles["productItem"]}>
                <div className={styles["productInfo"]}>
                  <span className={styles["productName"]}>
                    {product.name || product.product_name}
                  </span>
                  <span className={styles["quantity"]}>
                    x{product.quantity || 1}
                  </span>
                </div>
                {user && (
                  <button
                    className={styles["reviewBtn"]}
                    onClick={() => openReviewModal(product.product_id)}
                  >
                    ⭐ Đánh giá
                  </button>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Review Modal */}
      {showReviewModal && selectedProductId && (
        <div className={styles["modal"]}>
          <div className={styles["modalContent"]}>
            <ReviewForm
              productId={selectedProductId}
              userId={user?._id || user?.id}
              orderId={order.order_id}
              onSuccess={() => {
                setShowReviewModal(false);
              }}
              onClose={() => {
                setShowReviewModal(false);
              }}
            />
          </div>
        </div>
      )}

      {/* Shipping tracking removed - link intentionally omitted */}
    </div>
  );
};

export default OrderId;
