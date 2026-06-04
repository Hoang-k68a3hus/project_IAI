import React, { useEffect, useState } from "react";
import { useParams, useNavigate } from "react-router-dom";
import { useDispatch, useSelector } from "react-redux";
import { motion } from "framer-motion";
import styles from "./productDetail.module.css";
import { getProductById, getProductReviews, getSimilarProducts } from "../../api";
import { updateWishlist } from "../../actions/auth";
import Loading from "../../components/loading/Loading";
import ProductCard from "../../components/product-card/ProductCard";
import { getProductImageUrl } from "../../utils/productImages";

const ProductDetail = ({ addProductToCart }) => {
  const { id } = useParams();
  const navigate = useNavigate();
  const dispatch = useDispatch();
  
  const [product, setProduct] = useState(null);
  const [reviews, setReviews] = useState([]);
  const [reviewStats, setReviewStats] = useState(null);
  const [similarProducts, setSimilarProducts] = useState([]);
  const [loading, setLoading] = useState(true);
  const [reviewPage, setReviewPage] = useState(1);
  const [reviewPagination, setReviewPagination] = useState(null);
  const [quantity, setQuantity] = useState(1);
  const [activeTab, setActiveTab] = useState("description");
  const [sortBy, setSortBy] = useState("cmt_date");
  
  const wishlist = useSelector((state) => state.authentication.user?.wishlist) || [];

  // Format price
  const formatPrice = (price) => {
    if (!price) return "0 đ";
    return Number(price).toLocaleString("vi-VN") + " đ";
  };

  // Format date
  const formatDate = (date) => {
    if (!date) return "";
    return new Date(date).toLocaleDateString("vi-VN", {
      year: "numeric",
      month: "long",
      day: "numeric",
    });
  };

  const formatOptionalText = (value, fallback = "Không rõ") => {
    if (!value || String(value).startsWith("no_")) return fallback;
    return value;
  };

  // ⭐ Xử lý đường dẫn ảnh - thay .mp4 thành .jpg
  const getImageUrl = (imagePath) => {
    if (!imagePath) return getProductImageUrl(imagePath);
    const fixedPath = imagePath.replace(".mp4", ".jpg");
    return getProductImageUrl(fixedPath);
  };

  // Load product data
  useEffect(() => {
    const loadProduct = async () => {
      try {
        setLoading(true);
        const [productRes, reviewsRes, similarRes] = await Promise.all([
          getProductById(id),
          getProductReviews(id, 1, 10, sortBy),
          getSimilarProducts(id, 6),
        ]);
        
        setProduct(productRes.data);
        setReviews(reviewsRes.data.reviews);
        setReviewStats(reviewsRes.data.stats);
        setReviewPagination(reviewsRes.data.pagination);
        setSimilarProducts(similarRes.data);
      } catch (error) {
        console.error("Error loading product:", error);
        navigate("/404");
      } finally {
        setLoading(false);
      }
    };

    loadProduct();
  }, [id, navigate, sortBy]);

  // Load more reviews
  const loadMoreReviews = async () => {
    try {
      const nextPage = reviewPage + 1;
      const res = await getProductReviews(id, nextPage, 10, sortBy);
      setReviews([...reviews, ...res.data.reviews]);
      setReviewPage(nextPage);
      setReviewPagination(res.data.pagination);
    } catch (error) {
      console.error("Error loading reviews:", error);
    }
  };

  // Sort reviews
  const handleSortChange = async (newSortBy) => {
    setSortBy(newSortBy);
    try {
      const res = await getProductReviews(id, 1, 10, newSortBy);
      setReviews(res.data.reviews);
      setReviewPage(1);
      setReviewPagination(res.data.pagination);
    } catch (error) {
      console.error("Error sorting reviews:", error);
    }
  };

  // Add to wishlist
  const handleWishlist = () => {
    const onError = () => navigate("/login");
    dispatch(updateWishlist(product.product_id, onError));
  };

  // Add to cart
  const handleAddToCart = () => {
    for (let i = 0; i < quantity; i++) {
      addProductToCart(product);
    }
  };

  // Render stars
  const renderStars = (rating) => {
    const stars = [];
    const fullStars = Math.floor(rating);
    const hasHalfStar = rating % 1 >= 0.5;

    for (let i = 0; i < 5; i++) {
      if (i < fullStars) {
        stars.push(
          <span key={i} className={`${styles.star} ${styles.filled}`}>★</span>
        );
      } else if (i === fullStars && hasHalfStar) {
        stars.push(
          <span key={i} className={`${styles.star} ${styles.half}`}>★</span>
        );
      } else {
        stars.push(
          <span key={i} className={styles.star}>★</span>
        );
      }
    }
    return stars;
  };

  // Rating distribution bar
  const renderRatingBar = (starCount, count, total) => {
    const percentage = total > 0 ? (count / total) * 100 : 0;
    return (
      <div className={styles.ratingBar}>
        <span className={styles.starLabel}>{starCount} ★</span>
        <div className={styles.barContainer}>
          <div className={styles.barFill} style={{ width: `${percentage}%` }} />
        </div>
        <span className={styles.count}>{count}</span>
      </div>
    );
  };

  if (loading) {
    return <Loading />;
  }

  if (!product) {
    return <div className={styles.notFound}>Sản phẩm không tồn tại</div>;
  }

  // Get rating distribution
  const ratingDist = {};
  if (reviewStats?.ratingDistribution) {
    reviewStats.ratingDistribution.forEach((r) => {
      ratingDist[r._id] = r.count;
    });
  }

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className={styles.container}
    >
      {/* Breadcrumb */}
      <div className={styles.breadcrumb}>
        <span onClick={() => navigate("/")}>Trang chủ</span>
        <span className={styles.separator}>/</span>
        <span onClick={() => navigate("/products")}>Sản phẩm</span>
        <span className={styles.separator}>/</span>
        <span className={styles.current}>{product.name || product.product_name}</span>
      </div>

      {/* Product Main Section */}
      <div className={styles.productMain}>
        {/* Product Image */}
        <div className={styles.imageSection}>
          <img
            src={getImageUrl(product.image)}
            alt={product.name}
            className={styles.mainImage}
          />
        </div>

        {/* Product Info */}
        <div className={styles.infoSection}>
          <h1 className={styles.productName}>{product.name || product.product_name}</h1>
          
          <div className={styles.brandLine}>
            <span className={styles.brand}>Thương hiệu: <strong>{formatOptionalText(product.brand)}</strong></span>
          </div>

          <div className={styles.ratingLine}>
            <div className={styles.stars}>{renderStars(product.avg_rating || 0)}</div>
            <span className={styles.ratingValue}>{(product.avg_rating || 0).toFixed(1)}</span>
            <span className={styles.reviewCount}>({reviewPagination?.totalReviews || 0} đánh giá)</span>
            <span className={styles.soldCount}>Đã bán: {product.num_sold || 0}</span>
          </div>

          <div className={styles.priceSection}>
            <span className={styles.price}>{formatPrice(product.price)}</span>
          </div>

          {/* Product Attributes */}
          <div className={styles.attributes}>
            {product.skin_type && (
              <div className={styles.attribute}>
                <span className={styles.attrLabel}>Loại da:</span>
                <span className={styles.attrValue}>{formatOptionalText(product.skin_type)}</span>
              </div>
            )}
            {product.origin && product.origin !== "no_origin" && (
              <div className={styles.attribute}>
                <span className={styles.attrLabel}>Xuất xứ:</span>
                <span className={styles.attrValue}>{product.origin}</span>
              </div>
            )}
            {product.capacity && (
              <div className={styles.attribute}>
                <span className={styles.attrLabel}>Dung tích:</span>
                <span className={styles.attrValue}>{product.capacity}</span>
              </div>
            )}
            {product.type && product.type !== "no_type" && (
              <div className={styles.attribute}>
                <span className={styles.attrLabel}>Loại:</span>
                <span className={styles.attrValue}>{product.type}</span>
              </div>
            )}
          </div>

          {/* Quantity & Add to Cart */}
          <div className={styles.actionSection}>
            <div className={styles.quantityControl}>
              <button
                className={styles.qtyBtn}
                onClick={() => setQuantity(Math.max(1, quantity - 1))}
              >
                -
              </button>
              <span className={styles.quantity}>{quantity}</span>
              <button
                className={styles.qtyBtn}
                onClick={() => setQuantity(quantity + 1)}
              >
                +
              </button>
            </div>

            <button className={styles.addToCartBtn} onClick={handleAddToCart}>
              <span className="material-symbols-outlined">shopping_cart</span>
              Thêm vào giỏ hàng
            </button>

            <button
              className={`${styles.wishlistBtn} ${wishlist.includes(product.product_id) ? styles.wishlisted : ""}`}
              onClick={handleWishlist}
            >
              <span className="material-symbols-outlined">favorite</span>
            </button>
          </div>

          {/* Stock Status */}
          <div className={styles.stockStatus}>
            {product.stock > 0 ? (
              <span className={styles.inStock}>✓ Còn hàng ({product.stock} sản phẩm)</span>
            ) : (
              <span className={styles.outOfStock}>✕ Hết hàng</span>
            )}
          </div>
        </div>
      </div>

      {/* Tabs Section */}
      <div className={styles.tabsSection}>
        <div className={styles.tabHeaders}>
          <button
            className={`${styles.tabBtn} ${activeTab === "description" ? styles.active : ""}`}
            onClick={() => setActiveTab("description")}
          >
            Mô tả sản phẩm
          </button>
          <button
            className={`${styles.tabBtn} ${activeTab === "ingredients" ? styles.active : ""}`}
            onClick={() => setActiveTab("ingredients")}
          >
            Thành phần
          </button>
          <button
            className={`${styles.tabBtn} ${activeTab === "reviews" ? styles.active : ""}`}
            onClick={() => setActiveTab("reviews")}
          >
            Đánh giá ({reviewPagination?.totalReviews || 0})
          </button>
        </div>

        <div className={styles.tabContent}>
          {/* Description Tab */}
          {activeTab === "description" && (
            <div className={styles.descriptionTab}>
              <p>{product.description || product.processed_description || "Chưa có mô tả cho sản phẩm này."}</p>
              
              {product.feature && (
                <div className={styles.features}>
                  <h3>Đặc điểm nổi bật:</h3>
                  <p>{product.feature}</p>
                </div>
              )}
            </div>
          )}

          {/* Ingredients Tab */}
          {activeTab === "ingredients" && (
            <div className={styles.ingredientsTab}>
              {product.ingredient ? (
                <p>{product.ingredient}</p>
              ) : (
                <p>Chưa có thông tin thành phần.</p>
              )}
            </div>
          )}

          {/* Reviews Tab */}
          {activeTab === "reviews" && (
            <div className={styles.reviewsTab}>
              {/* Rating Summary */}
              <div className={styles.ratingSummary}>
                <div className={styles.overallRating}>
                  <span className={styles.bigRating}>
                    {(reviewStats?.averageRating || 0).toFixed(1)}
                  </span>
                  <div className={styles.stars}>{renderStars(reviewStats?.averageRating || 0)}</div>
                  <span className={styles.totalReviews}>
                    {reviewPagination?.totalReviews || 0} đánh giá
                  </span>
                </div>

                <div className={styles.ratingDistribution}>
                  {[5, 4, 3, 2, 1].map((star) => (
                    renderRatingBar(
                      star,
                      ratingDist[star] || 0,
                      reviewPagination?.totalReviews || 0
                    )
                  ))}
                </div>
              </div>

              {/* Sort Options */}
              <div className={styles.sortOptions}>
                <span>Sắp xếp theo:</span>
                <select value={sortBy} onChange={(e) => handleSortChange(e.target.value)}>
                  <option value="cmt_date">Mới nhất</option>
                  <option value="rating">Đánh giá cao nhất</option>
                </select>
              </div>

              {/* Reviews List */}
              <div className={styles.reviewsList}>
                {reviews.length > 0 ? (
                  reviews.map((review, index) => (
                    <div key={index} className={styles.reviewItem}>
                      <div className={styles.reviewHeader}>
                        <div className={styles.reviewerInfo}>
                          <span className={styles.reviewerAvatar}>
                            {String(review.user_id).slice(0, 2)}
                          </span>
                          <span className={styles.reviewerId}>
                            Người dùng #{review.user_id}
                          </span>
                        </div>
                        <div className={styles.reviewMeta}>
                          <div className={styles.reviewStars}>
                            {renderStars(review.rating)}
                          </div>
                          <span className={styles.reviewDate}>
                            {formatDate(review.cmt_date)}
                          </span>
                        </div>
                      </div>
                      
                      {review.variation && (
                        <div className={styles.reviewVariation}>
                          Phân loại: {review.variation}
                        </div>
                      )}
                      
                      <div className={styles.reviewContent}>
                        {review.comment || review.processed_comment || "Không có nội dung đánh giá."}
                      </div>
                      
                      {review.product_quality && (
                        <div className={styles.qualityRating}>
                          Chất lượng sản phẩm: {review.product_quality}/5
                        </div>
                      )}
                    </div>
                  ))
                ) : (
                  <div className={styles.noReviews}>
                    Chưa có đánh giá nào cho sản phẩm này.
                  </div>
                )}
              </div>

              {/* Load More */}
              {reviewPagination && reviewPage < reviewPagination.totalPages && (
                <button className={styles.loadMoreBtn} onClick={loadMoreReviews}>
                  Xem thêm đánh giá
                </button>
              )}
            </div>
          )}
        </div>
      </div>

      {/* Similar Products */}
      {similarProducts.length > 0 && (
        <div className={styles.similarSection}>
          <h2 className={styles.sectionTitle}>Sản phẩm tương tự</h2>
          <div className={styles.similarGrid}>
            {similarProducts.map((prod) => (
              <ProductCard
                key={prod.product_id}
                product={prod}
                addProductToCart={addProductToCart}
              />
            ))}
          </div>
        </div>
      )}
    </motion.div>
  );
};

export default ProductDetail;
