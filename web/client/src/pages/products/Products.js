import styles from "./products.module.css";
import { useState, useEffect } from "react";
import { useLocation, useNavigate } from "react-router-dom";
import { useDispatch } from "react-redux";
import { getProductsPerPage } from "../../actions/products";
import ProductCard from "../../components/product-card/ProductCard";
import Pages from "../../components/pages/Pages";
import Loading from "../../components/loading/Loading";
import { getRecommendations, semanticSearch } from "../../api/viecomrec";
import axios from "axios";
import { PRODUCTS_BASEURL } from "../../api/BaseURLs";

const Products = ({ addProductToCart }) => {
  const [page, setPage] = useState(1);
  const [products, setProducts] = useState([]);
  const [totalPages, setTotalPages] = useState(1);
  const [loading, setLoading] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [isSemanticSearch, setIsSemanticSearch] = useState(false);

  // 👉 NEW: Recommendation state
  const [recommendations, setRecommendations] = useState([]);

  const location = useLocation();
  const dispatch = useDispatch();
  const navigate = useNavigate();

  useEffect(() => {
    setLoading(true);
    const query = new URLSearchParams(location.search);
    const currentPage = query.get("page") || 1;
    setPage(Number(currentPage));

    if (query.get("search")) {
      const search = query.get("search");
      setSearchQuery(search);
      // 👉 Sử dụng VieComRec semantic search
      handleSemanticSearch(search, Number(currentPage));
    } else if (query.get("category")) {
      setSearchQuery("");
      setIsSemanticSearch(false);
      const category = query.get("category");
      if (category === "All") {
        dispatch(getProductsPerPage(currentPage, null, onSuccess));
      } else {
        dispatch(getProductsPerPage(currentPage, category, onSuccess));
      }
    } else {
      setSearchQuery("");
      setIsSemanticSearch(false);
      dispatch(getProductsPerPage(currentPage, null, onSuccess));
    }
    // This effect is keyed to URL changes; the helpers only resolve that URL state.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dispatch, location.search]);

  // 👉 VieComRec Semantic Search
  const handleSemanticSearch = async (query, currentPage) => {
    try {
      const result = await semanticSearch(query, 20);

      if (result && result.results) {
        setIsSemanticSearch(true);
        
        // Lấy danh sách product_ids từ search results
        const productIds = result.results.map((item) => item.product_id);
        
        // Gọi API server để lấy đầy đủ thông tin sản phẩm từ MongoDB (bao gồm image)
        const productsRes = await axios.post(`${PRODUCTS_BASEURL}/arr`, {
          arr: productIds,
        });
        
        // Tạo map để tra cứu nhanh product info
        const productMap = {};
        productsRes.data.forEach((p) => {
          productMap[p.product_id] = p;
        });
        
        // Merge thông tin từ VieComRec với MongoDB
        const mapped = result.results.map((item) => {
          const dbProduct = productMap[item.product_id] || {};
          return {
            product_id: item.product_id,
            name: dbProduct.name || dbProduct.product_name || item.product_name,
            price: dbProduct.price || item.price,
            brand: dbProduct.brand || item.brand,
            category: dbProduct.category || item.category,
            avg_rating: dbProduct.avg_rating || item.avg_rating,
            num_sold: dbProduct.num_sold || item.num_sold,
            semantic_score: item.semantic_score,
            final_score: item.final_score,
            // Lấy image từ MongoDB
            image: dbProduct.image || `/images/products/${item.product_id}.jpg`,
          };
        });

        // Pagination cho kết quả semantic search
        const itemsPerPage = 12;
        const startIndex = (currentPage - 1) * itemsPerPage;
        const paginatedProducts = mapped.slice(startIndex, startIndex + itemsPerPage);
        
        setProducts(paginatedProducts);
        setTotalPages(Math.ceil(mapped.length / itemsPerPage));
        setLoading(false);
        loadRecommendations();
      } else {
        // Fallback nếu VieComRec không hoạt động
        setIsSemanticSearch(false);
        setProducts([]);
        setTotalPages(1);
        setLoading(false);
      }
    } catch (err) {
      console.error("Semantic search error:", err);
      setIsSemanticSearch(false);
      setProducts([]);
      setTotalPages(1);
      setLoading(false);
    }
  };

  const onSuccess = (res) => {
    setTotalPages(res.total_pages);
    setProducts(res.products);
    setLoading(false);

    // 👉 Load recommendations ngay khi load sản phẩm thành công
    loadRecommendations();
  };

  // 👉 NEW: API gọi VieComRec recommend
  const loadRecommendations = async () => {
    try {
      // Lấy user_id từ localStorage nếu đã đăng nhập
      const profile = JSON.parse(localStorage.getItem("profile"));
      const userId = profile?.user?.user_id || 1;

      const result = await getRecommendations(userId, 8, true);

      if (result && result.recommendations) {
        // Lấy danh sách product_ids từ recommendations
        const productIds = result.recommendations.map((item) => item.product_id);
        
        // Gọi API server để lấy đầy đủ thông tin sản phẩm từ MongoDB
        const productsRes = await axios.post(`${PRODUCTS_BASEURL}/arr`, {
          arr: productIds,
        });
        
        // Tạo map để tra cứu nhanh product info
        const productMap = {};
        productsRes.data.forEach((p) => {
          productMap[p.product_id] = p;
        });
        
        // Merge thông tin từ VieComRec với MongoDB
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
            // Lấy image từ MongoDB
            image: dbProduct.image || `/images/products/${item.product_id}.jpg`,
          };
        });
        setRecommendations(mapped);
      }
    } catch (err) {
      console.error("Load recommendations error:", err);
    }
  };

  const handleClick = (i) => {
    const query = new URLSearchParams(location.search);

    if (query.get("search")) {
      const search = query.get("search");
      navigate(`/products?search=${search}&page=${i}`);
    } else if (query.get("category")) {
      const category = query.get("category");
      navigate(`/products?category=${category}&page=${i}`);
    } else {
      navigate(`/products?page=${i}`);
    }

    window.scrollTo(0, 0);
  };

  return (
    <div className={styles["wrapper"]}>
      <div className={"heading"}>
        <h1>
          {searchQuery 
            ? `Kết quả tìm kiếm: "${searchQuery}"` 
            : "Products"}
        </h1>
        {isSemanticSearch && (
          <p style={{ fontSize: "0.9rem", color: "#666", marginTop: "0.5rem" }}>
            🔍 Tìm kiếm thông minh bằng AI
          </p>
        )}
      </div>

      {loading ? (
        <Loading />
      ) : (
        <>
          {/* PRODUCT LIST */}
          {products.length > 0 ? (
            <div className={styles["products-wrapper"]}>
              {products.map((product, i) => (
                <ProductCard
                  key={i}
                  product={product}
                  productsPage={true}
                  addProductToCart={addProductToCart}
                />
              ))}
            </div>
          ) : (
            <div style={{ textAlign: "center", padding: "2rem", color: "#666" }}>
              <p>Không tìm thấy sản phẩm nào</p>
              {searchQuery && (
                <p style={{ fontSize: "0.9rem", marginTop: "0.5rem" }}>
                  Thử tìm kiếm với từ khóa khác
                </p>
              )}
            </div>
          )}

          {/* PAGINATION */}
          <Pages max={totalPages} current={page} onPageClick={handleClick} />

          {/* 👉 NEW: RECOMMENDED SECTION */}
          {recommendations.length > 0 && (
            <section style={{ marginTop: "4rem" }}>
              <div className={"heading-wrapper"}>
                <h1 className={"heading"}>Gợi ý dành cho bạn</h1>
              </div>

              <div className={styles["products-wrapper"]}>
                {recommendations.map((item, index) => (
                  <ProductCard
                    key={`rec-${index}`}
                    product={item}
                    addProductToCart={addProductToCart}
                  />
                ))}
              </div>
            </section>
          )}
        </>
      )}
    </div>
  );
};

export default Products;
