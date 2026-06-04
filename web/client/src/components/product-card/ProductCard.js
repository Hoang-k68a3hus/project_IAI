import styles from "./productCard.module.css";
import { useRef, useState } from "react";
import { motion } from "framer-motion";
import { useDispatch, useSelector } from "react-redux";
import { updateWishlist } from "../../actions/auth";
import { useNavigate } from "react-router-dom";
import { getProductImageUrl } from "../../utils/productImages";

const ProductCard = ({ product, addProductToCart, productsPage = false }) => {
  const [addToCart, setAddToCart] = useState(false);
  const wrapperRef = useRef();
  const wishlist =
    useSelector((state) => state.authentication.user?.wishlist) || [];
  const dispatch = useDispatch();
  const navigate = useNavigate();

  const formatPrice = (price) => {
    if (!price) return "0 đ";
    return Number(price).toLocaleString("vi-VN") + " đ";
  };

  const formatOptionalText = (value) => {
    if (!value || String(value).startsWith("no_")) return "";
    return value;
  };

  const getImageUrl = (imagePath) => {
    if (!imagePath) return getProductImageUrl(imagePath);
    return getProductImageUrl(String(imagePath).replace(".mp4", ".jpg"));
  };

  const rating = Number(product.avg_rating || product.avg_star || 0);
  const sold = Number(product.num_sold || product.num_sold_time || 0);
  const label = formatOptionalText(product.brand) || formatOptionalText(product.category);

  const handleWishlist = () => {
    const onError = () => {
      navigate("/login");
    };

    dispatch(updateWishlist(product.product_id, onError));
  };

  const handleAddToCart = () => {
    setAddToCart(true);

    setTimeout(() => {
      setAddToCart(false);
      addProductToCart(product);
    }, 600);
  };

  const getXi = () => wrapperRef.current.getBoundingClientRect().x;

  const getXf = () => {
    const windowWidth = window.innerWidth;
    if (windowWidth > 1024) return windowWidth - 11 * 16;
    return windowWidth - 5 * 16;
  };

  const getYi = () => wrapperRef.current.getBoundingClientRect().y;

  return (
    <div
      ref={wrapperRef}
      className={`${styles["wrapper"]} ${
        productsPage ? styles["products-page"] : ""
      }`}
    >
      {addToCart && (
        <motion.img
          initial={{
            x: getXi(),
            y: getYi(),
            padding: "1em",
            borderRadius: "8px",
          }}
          animate={{
            x: getXf(),
            y: 0,
            width: 24,
            height: 24,
            opacity: 0.8,
            borderRadius: "50%",
            padding: ".5em",
          }}
          transition={{ type: "spring", stiffness: 40, bounce: 0 }}
          className={styles["cart-img"]}
          src={getImageUrl(product.image)}
          alt={product.name}
        />
      )}

      <div
        className={styles["image-wrapper"]}
        onClick={() => navigate(`/products/${product.product_id}`)}
        style={{ cursor: "pointer" }}
      >
        <img src={getImageUrl(product.image)} alt={product.name} />
        {label && <span className={styles["label"]}>{label}</span>}

        <span
          onClick={(e) => {
            e.stopPropagation();
            handleWishlist();
          }}
          className={`material-symbols-outlined ${styles["wishlist"]} ${
            wishlist.includes(product.product_id) && styles["wishlisted"]
          }`}
        >
          favorite
        </span>
      </div>

      <div className={styles["content"]}>
        <p
          className={styles["name"]}
          onClick={() => navigate(`/products/${product.product_id}`)}
          style={{ cursor: "pointer" }}
        >
          {product.name || product.product_name}
        </p>

        <div className={styles["footer"]}>
          <div className={styles["meta"]}>
            <span>
              <span className="material-symbols-outlined">star</span>
              {rating ? rating.toFixed(1) : "Mới"}
            </span>
            <span>{sold ? `${sold.toLocaleString("vi-VN")} đã bán` : "Sẵn hàng"}</span>
          </div>

          <div className={styles["details"]}>
            <p className={styles["weight"]}>
              {formatOptionalText(product.category) || "Chăm sóc da"}
            </p>
            <p className={styles["price"]}>{formatPrice(product.price)}</p>
          </div>

          <button onClick={handleAddToCart} className={styles["add-to-cart"]}>
            <span className="material-symbols-outlined">add_shopping_cart</span>
            Thêm giỏ
          </button>
        </div>
      </div>
    </div>
  );
};

export default ProductCard;
