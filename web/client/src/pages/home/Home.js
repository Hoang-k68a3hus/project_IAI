import styles from "./home.module.css";
import { Link } from "react-router-dom";
import ProductCard from "../../components/product-card/ProductCard";
import { motion } from "framer-motion";
import { useCallback, useEffect, useState } from "react";
import Loading from "../../components/loading/Loading";
import axios from "axios";
import { getRecommendations } from "../../api/viecomrec";
import { PRODUCTS_BASEURL } from "../../api/BaseURLs";
import { HERO_DELIVERY_IMAGE, WHY_IMAGES } from "../../shared/placeholders";

const Home = ({ addProductToCart }) => {
  const [products, setProducts] = useState([]);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [hasMore, setHasMore] = useState(true);

  // 👉 Recommendation state
  const [recommendations, setRecommendations] = useState([]);
  const [recLoading, setRecLoading] = useState(false);

  const loadProducts = useCallback(async () => {
    try {
      const res = await axios.get(
        `${PRODUCTS_BASEURL}?page=${page}&limit=12`
      );

      const newProducts = res.data?.products || [];

      if (newProducts.length === 0) {
        setHasMore(false);
      } else {
        setProducts((prev) => [...prev, ...newProducts]);
      }

      setLoading(false);
      setLoadingMore(false);
    } catch (error) {
      console.error("Load products error:", error);
      setLoading(false);
      setLoadingMore(false);
    }
  }, [page]);

  // 👉 Load recommendations từ VieComRec API
  const loadRecommendations = useCallback(async () => {
    setRecLoading(true);
    try {
      // Lấy user_id từ localStorage nếu đã đăng nhập, không thì dùng 1
      const profile = JSON.parse(localStorage.getItem("profile"));
      const userId = profile?.user?.user_id || 1;

      const result = await getRecommendations(userId, 10, true);

      if (result && result.recommendations) {
        // Lấy danh sách product_ids từ recommendations
        const productIds = result.recommendations.map((item) => item.product_id);
        
        // Gọi API server để lấy đầy đủ thông tin sản phẩm từ MongoDB (bao gồm image)
        const productsRes = await axios.post(`${PRODUCTS_BASEURL}/arr`, {
          arr: productIds,
        });
        
        // Tạo map để tra cứu nhanh product info
        const productMap = {};
        productsRes.data.forEach((p) => {
          productMap[p.product_id] = p;
        });
        
        // Merge thông tin từ VieComRec với MongoDB, giữ thứ tự từ recommendations
        const mapped = result.recommendations.map((item) => {
          const dbProduct = productMap[item.product_id] || {};
          return {
            product_id: item.product_id,
            name: dbProduct.name || dbProduct.product_name || item.product_name,
            price: dbProduct.price || item.price,
            brand: dbProduct.brand || item.brand,
            category: dbProduct.category || item.category,
            avg_rating: dbProduct.avg_rating || item.avg_rating,
            num_sold: dbProduct.num_sold || item.num_sold,
            score: item.score,
            // Lấy image từ MongoDB (đã có đường dẫn đầy đủ)
            image: dbProduct.image || `/images/products/${item.product_id}.jpg`,
          };
        });
        setRecommendations(mapped);
      }
    } catch (err) {
      console.error("Load recommendations error:", err);
    } finally {
      setRecLoading(false);
    }
  }, []);

  useEffect(() => {
    loadProducts();
    loadRecommendations();
  }, [loadProducts, loadRecommendations]);

  useEffect(() => {
    const handleScroll = () => {
      if (
        window.innerHeight + window.scrollY >=
          document.body.offsetHeight - 300 &&
        !loadingMore &&
        hasMore
      ) {
        setLoadingMore(true);
        setPage((prev) => prev + 1);
      }
    };

    window.addEventListener("scroll", handleScroll);
    return () => window.removeEventListener("scroll", handleScroll);
  }, [loadingMore, hasMore]);

  if (loading) return <Loading />;

  return (
    <div className={styles["wrapper"]}>
      {/* HERO */}
      <div className={styles["hero"]}>
        <div className={styles["hero-text"]}>
          <h1>Mỹ phẩm chính hãng giao nhanh</h1>
          <p>Mua sắm tiện lợi – Giao trong 20 phút</p>
          <Link className={"btn1"} to={"products"}>
            Shop Now
          </Link>
        </div>

        <div className={styles["delivery"]}>
          <motion.img
            drag
            dragConstraints={{ top: 0, right: 0, bottom: 0, left: 0 }}
            src={HERO_DELIVERY_IMAGE}
            alt="delivery"
          />
        </div>
      </div>

      {/* FEATURED PRODUCTS */}
      <section>
        <div className={"heading-wrapper"}>
          <h1 className={"heading"}>Sản phẩm nổi bật</h1>
        </div>

        <div className={styles["products-wrapper"]}>
          {products.map((product, index) => (
            <ProductCard
              key={index}
              product={product}
              addProductToCart={addProductToCart}
            />
          ))}
        </div>

        {loadingMore && (
          <p style={{ textAlign: "center", padding: "1em", color: "#666" }}>
            Đang tải thêm...
          </p>
        )}
      </section>

      {/* RECOMMENDED FOR YOU */}
      {recommendations.length > 0 && (
        <section>
          <div className={"heading-wrapper"}>
            <h1 className={"heading"}>Gợi ý dành cho bạn</h1>
          </div>

          {recLoading ? (
            <p style={{ textAlign: "center", padding: "1em", color: "#666" }}>
              Đang tải gợi ý...
            </p>
          ) : (
            <div className={styles["products-wrapper"]}>
              {recommendations.map((product, index) => (
                <ProductCard
                  key={`rec-${index}`}
                  product={product}
                  addProductToCart={addProductToCart}
                />
              ))}
            </div>
          )}
        </section>
      )}

      {/* WHY */}
      <div className={styles["why"]}>
        <div className={styles["why-component"]}>
          <img src={WHY_IMAGES.delivery} alt={"Delivery"} />
          <div className={"why-text"}>
            <div className={styles["why-title"]}>Giao nhanh</div>
            <div className={styles["why-desc"]}>Nhanh – đúng giờ – an tâm</div>
          </div>
        </div>

        <div className={styles["why-component"]}>
          <img src={WHY_IMAGES.reliable} alt={"Reliable"} />
          <div className={"why-text"}>
            <div className={styles["why-title"]}>Uy tín</div>
            <div className={styles["why-desc"]}>Hàng chuẩn – Hàng thật</div>
          </div>
        </div>

        <div className={styles["why-component"]}>
          <img src={WHY_IMAGES.prices} alt={"Prices"} />
          <div className={"why-text"}>
            <div className={styles["why-title"]}>Giá tốt</div>
            <div className={styles["why-desc"]}>Khuyến mãi mỗi ngày</div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Home;
