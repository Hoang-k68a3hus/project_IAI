import React, { useEffect, useState } from "react";
import styles from "./aiDashboard.module.css";
import {
  checkHealth,
  getModelInfo,
  semanticSearch,
  getRecommendations,
  getSimilarItems,
} from "../../../api/viecomrec";
import axios from "axios";
import {
  PRODUCTS_BASEURL,
  STREAMLIT_BASEURL,
  VIECOMREC_BASEURL,
} from "../../../api/BaseURLs";
import ProductCard from "../../../components/product-card/ProductCard";

const VIECOMREC_URL = VIECOMREC_BASEURL;
const STREAMLIT_URL = STREAMLIT_BASEURL;

const AIDashboard = ({ addProductToCart }) => {
  // Tab state
  const [activeTab, setActiveTab] = useState("demo");

  // Health & Model Info
  const [health, setHealth] = useState(null);
  const [modelInfo, setModelInfo] = useState(null);
  const [loading, setLoading] = useState(true);

  // Demo state
  const [demoTab, setDemoTab] = useState("recommend");
  const [userId, setUserId] = useState(14);
  const [searchQuery, setSearchQuery] = useState("");
  const [productId, setProductId] = useState("");
  const [topK, setTopK] = useState(10);

  // Results
  const [recommendations, setRecommendations] = useState([]);
  const [searchResults, setSearchResults] = useState([]);
  const [similarItems, setSimilarItems] = useState([]);
  const [resultInfo, setResultInfo] = useState(null);
  const [demoLoading, setDemoLoading] = useState(false);

  // Filters
  const [filters, setFilters] = useState(null);
  const [selectedBrand, setSelectedBrand] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("");
  const [minPrice, setMinPrice] = useState("");
  const [maxPrice, setMaxPrice] = useState("");

  // Streamlit iframe
  const [streamlitLoaded, setStreamlitLoaded] = useState(false);

  useEffect(() => {
    loadInitialData();
  }, []);

  const loadInitialData = async () => {
    setLoading(true);
    try {
      const [healthData, modelData, filtersRes] = await Promise.all([
        checkHealth(),
        getModelInfo(),
        axios.get(`${VIECOMREC_URL}/search/filters`).catch(() => ({ data: null })),
      ]);
      setHealth(healthData);
      setModelInfo(modelData);
      setFilters(filtersRes.data);
    } catch (err) {
      console.error("Error loading data:", err);
    } finally {
      setLoading(false);
    }
  };

  // Fetch product details from MongoDB
  const fetchProductDetails = async (productIds) => {
    try {
      const res = await axios.post(`${PRODUCTS_BASEURL}/arr`, { arr: productIds });
      const productMap = {};
      res.data.forEach((p) => {
        productMap[p.product_id] = p;
      });
      return productMap;
    } catch (err) {
      console.error("Error fetching product details:", err);
      return {};
    }
  };

  // Demo: Get Recommendations
  const handleGetRecommendations = async () => {
    setDemoLoading(true);
    try {
      const filterParams = {};
      if (selectedBrand) filterParams.brand = selectedBrand;
      if (selectedCategory) filterParams.category = selectedCategory;
      if (minPrice) filterParams.min_price = parseInt(minPrice);
      if (maxPrice) filterParams.max_price = parseInt(maxPrice);

      const result = await getRecommendations(
        userId,
        topK,
        true,
        Object.keys(filterParams).length > 0 ? filterParams : null
      );

      if (result?.recommendations) {
        const productIds = result.recommendations.map((r) => r.product_id);
        const productMap = await fetchProductDetails(productIds);

        const mapped = result.recommendations.map((item) => {
          const dbProduct = productMap[item.product_id] || {};
          return {
            ...item,
            name: dbProduct.name || dbProduct.product_name || item.product_name,
            image: dbProduct.image || `/images/products/${item.product_id}.jpg`,
          };
        });

        setRecommendations(mapped);
        setResultInfo({
          type: "recommend",
          user_id: result.user_id,
          count: result.count,
          is_fallback: result.is_fallback,
          fallback_method: result.fallback_method,
          latency_ms: result.latency_ms,
          model_id: result.model_id,
        });
      }
    } catch (err) {
      console.error("Recommendation error:", err);
    } finally {
      setDemoLoading(false);
    }
  };

  // Demo: Semantic Search
  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    setDemoLoading(true);
    try {
      const filterParams = {};
      if (selectedBrand) filterParams.brand = selectedBrand;
      if (selectedCategory) filterParams.category = selectedCategory;
      if (minPrice) filterParams.min_price = parseInt(minPrice);
      if (maxPrice) filterParams.max_price = parseInt(maxPrice);

      const result = await semanticSearch(
        searchQuery,
        topK,
        Object.keys(filterParams).length > 0 ? filterParams : null
      );

      if (result?.results) {
        const productIds = result.results.map((r) => r.product_id);
        const productMap = await fetchProductDetails(productIds);

        const mapped = result.results.map((item) => {
          const dbProduct = productMap[item.product_id] || {};
          return {
            ...item,
            name: dbProduct.name || dbProduct.product_name || item.product_name,
            image: dbProduct.image || `/images/products/${item.product_id}.jpg`,
          };
        });

        setSearchResults(mapped);
        setResultInfo({
          type: "search",
          query: result.query,
          count: result.count,
          method: result.method,
          latency_ms: result.latency_ms,
        });
      }
    } catch (err) {
      console.error("Search error:", err);
    } finally {
      setDemoLoading(false);
    }
  };

  // Demo: Similar Items
  const handleSimilarItems = async () => {
    if (!productId) return;
    setDemoLoading(true);
    try {
      const result = await getSimilarItems(parseInt(productId), topK);

      if (result?.similar_items) {
        const productIds = result.similar_items.map((r) => r.product_id);
        const productMap = await fetchProductDetails(productIds);

        const mapped = result.similar_items.map((item) => {
          const dbProduct = productMap[item.product_id] || {};
          return {
            ...item,
            name: dbProduct.name || dbProduct.product_name || `Product #${item.product_id}`,
            price: dbProduct.price,
            brand: dbProduct.brand,
            image: dbProduct.image || `/images/products/${item.product_id}.jpg`,
          };
        });

        setSimilarItems(mapped);
        setResultInfo({
          type: "similar",
          product_id: result.product_id,
          count: result.count,
          method: result.method,
        });
      }
    } catch (err) {
      console.error("Similar items error:", err);
    } finally {
      setDemoLoading(false);
    }
  };

  const formatNumber = (num) => {
    if (!num) return "0";
    return Number(num).toLocaleString("vi-VN");
  };

  const clearFilters = () => {
    setSelectedBrand("");
    setSelectedCategory("");
    setMinPrice("");
    setMaxPrice("");
  };

  if (loading) {
    return (
      <div className={styles.container}>
        <div className={styles.loading}>
          <div className={styles.spinner}></div>
          <p>Đang kết nối VieComRec API...</p>
        </div>
      </div>
    );
  }

  return (
    <div className={styles.container}>
      {/* Header */}
      <div className={styles.header}>
        <div className={styles.headerContent}>
          <h1>🤖 VieComRec AI Demo</h1>
          <p>Hệ thống gợi ý mỹ phẩm thông minh - PhoBERT + ALS Hybrid Model</p>
        </div>
        <div className={styles.headerStats}>
          <div className={`${styles.statusBadge} ${health?.status === "healthy" ? styles.healthy : styles.error}`}>
            {health?.status === "healthy" ? "✅ API Online" : "❌ API Offline"}
          </div>
        </div>
      </div>

      {/* Status Cards */}
      <div className={styles.statusCards}>
        <div className={styles.statusCard}>
          <div className={styles.statusIcon}>🧠</div>
          <div className={styles.statusInfo}>
            <h3>Model</h3>
            <p>{health?.model_type || "N/A"}</p>
            <small>{modelInfo?.factors} factors</small>
          </div>
        </div>

        <div className={styles.statusCard}>
          <div className={styles.statusIcon}>👥</div>
          <div className={styles.statusInfo}>
            <h3>Users</h3>
            <p>{formatNumber(health?.num_users)}</p>
            <small>Trainable: {formatNumber(health?.trainable_users)}</small>
          </div>
        </div>

        <div className={styles.statusCard}>
          <div className={styles.statusIcon}>📦</div>
          <div className={styles.statusInfo}>
            <h3>Products</h3>
            <p>{formatNumber(health?.num_items)}</p>
            <small>{filters?.brands?.length || 0} brands</small>
          </div>
        </div>

        <div className={styles.statusCard}>
          <div className={styles.statusIcon}>🔤</div>
          <div className={styles.statusInfo}>
            <h3>Search</h3>
            <p>PhoBERT</p>
            <small>Semantic Vietnamese</small>
          </div>
        </div>
      </div>

      {/* Main Tabs */}
      <div className={styles.mainTabs}>
        <button
          className={`${styles.mainTab} ${activeTab === "demo" ? styles.active : ""}`}
          onClick={() => setActiveTab("demo")}
        >
          🎮 Demo API
        </button>
        <button
          className={`${styles.mainTab} ${activeTab === "dashboard" ? styles.active : ""}`}
          onClick={() => setActiveTab("dashboard")}
        >
          📊 Full Dashboard
        </button>
        <button
          className={`${styles.mainTab} ${activeTab === "docs" ? styles.active : ""}`}
          onClick={() => setActiveTab("docs")}
        >
          📚 API Docs
        </button>
      </div>

      {/* Tab Content */}
      <div className={styles.tabContent}>
        {/* Demo Tab */}
        {activeTab === "demo" && (
          <div className={styles.demoSection}>
            {/* Demo Sub-tabs */}
            <div className={styles.demoTabs}>
              <button
                className={`${styles.demoTab} ${demoTab === "recommend" ? styles.active : ""}`}
                onClick={() => setDemoTab("recommend")}
              >
                🎯 Gợi ý cho User
              </button>
              <button
                className={`${styles.demoTab} ${demoTab === "search" ? styles.active : ""}`}
                onClick={() => setDemoTab("search")}
              >
                🔍 Tìm kiếm AI
              </button>
              <button
                className={`${styles.demoTab} ${demoTab === "similar" ? styles.active : ""}`}
                onClick={() => setDemoTab("similar")}
              >
                🔗 Sản phẩm tương tự
              </button>
            </div>

            {/* Filters */}
            <div className={styles.filtersSection}>
              <h3>⚙️ Filters</h3>
              <div className={styles.filtersGrid}>
                <div className={styles.filterItem}>
                  <label>Brand</label>
                  <select value={selectedBrand} onChange={(e) => setSelectedBrand(e.target.value)}>
                    <option value="">Tất cả</option>
                    {filters?.brands?.slice(0, 50).map((b) => (
                      <option key={b} value={b}>{b}</option>
                    ))}
                  </select>
                </div>
                <div className={styles.filterItem}>
                  <label>Category</label>
                  <select value={selectedCategory} onChange={(e) => setSelectedCategory(e.target.value)}>
                    <option value="">Tất cả</option>
                    {filters?.categories?.map((c) => (
                      <option key={c} value={c}>{c}</option>
                    ))}
                  </select>
                </div>
                <div className={styles.filterItem}>
                  <label>Giá từ</label>
                  <input
                    type="number"
                    value={minPrice}
                    onChange={(e) => setMinPrice(e.target.value)}
                    placeholder="VNĐ"
                  />
                </div>
                <div className={styles.filterItem}>
                  <label>Giá đến</label>
                  <input
                    type="number"
                    value={maxPrice}
                    onChange={(e) => setMaxPrice(e.target.value)}
                    placeholder="VNĐ"
                  />
                </div>
                <div className={styles.filterItem}>
                  <label>Top K</label>
                  <input
                    type="number"
                    value={topK}
                    onChange={(e) => setTopK(parseInt(e.target.value) || 10)}
                    min="1"
                    max="50"
                  />
                </div>
                <button className={styles.clearBtn} onClick={clearFilters}>
                  🗑️ Clear
                </button>
              </div>
            </div>

            {/* Recommend Demo */}
            {demoTab === "recommend" && (
              <div className={styles.demoPanel}>
                <h3>🎯 POST /recommend</h3>
                <p>Gợi ý sản phẩm dựa trên hành vi người dùng (ALS Collaborative Filtering)</p>
                
                <div className={styles.inputRow}>
                  <div className={styles.inputGroup}>
                    <label>User ID</label>
                    <input
                      type="number"
                      value={userId}
                      onChange={(e) => setUserId(parseInt(e.target.value) || 1)}
                      placeholder="Nhập User ID"
                    />
                  </div>
                  <button
                    className={styles.runBtn}
                    onClick={handleGetRecommendations}
                    disabled={demoLoading}
                  >
                    {demoLoading ? "⏳ Loading..." : "▶️ Chạy Demo"}
                  </button>
                </div>

                <div className={styles.codeExample}>
                  <pre>{`curl -X POST ${VIECOMREC_URL}/recommend \\
  -H "Content-Type: application/json" \\
  -d '{"user_id": ${userId}, "topk": ${topK}${selectedBrand ? `, "filter_params": {"brand": "${selectedBrand}"}` : ""}}'`}</pre>
                </div>
              </div>
            )}

            {/* Search Demo */}
            {demoTab === "search" && (
              <div className={styles.demoPanel}>
                <h3>🔍 POST /search</h3>
                <p>Tìm kiếm ngữ nghĩa tiếng Việt với PhoBERT embeddings</p>
                
                <div className={styles.inputRow}>
                  <div className={styles.inputGroup}>
                    <label>Query</label>
                    <input
                      type="text"
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      placeholder="VD: kem dưỡng ẩm cho da khô, serum vitamin C..."
                      onKeyPress={(e) => e.key === "Enter" && handleSearch()}
                    />
                  </div>
                  <button
                    className={styles.runBtn}
                    onClick={handleSearch}
                    disabled={demoLoading || !searchQuery.trim()}
                  >
                    {demoLoading ? "⏳ Loading..." : "🔍 Tìm kiếm"}
                  </button>
                </div>

                <div className={styles.codeExample}>
                  <pre>{`curl -X POST ${VIECOMREC_URL}/search \\
  -H "Content-Type: application/json" \\
  -d '{"query": "${searchQuery || "kem dưỡng ẩm"}", "topk": ${topK}}'`}</pre>
                </div>
              </div>
            )}

            {/* Similar Items Demo */}
            {demoTab === "similar" && (
              <div className={styles.demoPanel}>
                <h3>🔗 POST /similar_items</h3>
                <p>Tìm sản phẩm tương tự dựa trên CF item-item similarity</p>
                
                <div className={styles.inputRow}>
                  <div className={styles.inputGroup}>
                    <label>Product ID</label>
                    <input
                      type="number"
                      value={productId}
                      onChange={(e) => setProductId(e.target.value)}
                      placeholder="Nhập Product ID"
                    />
                  </div>
                  <button
                    className={styles.runBtn}
                    onClick={handleSimilarItems}
                    disabled={demoLoading || !productId}
                  >
                    {demoLoading ? "⏳ Loading..." : "🔗 Tìm tương tự"}
                  </button>
                </div>

                <div className={styles.codeExample}>
                  <pre>{`curl -X POST ${VIECOMREC_URL}/similar_items \\
  -H "Content-Type: application/json" \\
  -d '{"product_id": ${productId || 125899}, "topk": ${topK}}'`}</pre>
                </div>
              </div>
            )}

            {/* Result Info */}
            {resultInfo && (
              <div className={styles.resultInfo}>
                <h4>📊 Kết quả API</h4>
                <div className={styles.resultMeta}>
                  {resultInfo.type === "recommend" && (
                    <>
                      <span>User: {resultInfo.user_id}</span>
                      <span>Count: {resultInfo.count}</span>
                      <span className={resultInfo.is_fallback ? styles.fallback : styles.cf}>
                        {resultInfo.is_fallback ? `Fallback: ${resultInfo.fallback_method}` : "CF Model"}
                      </span>
                      <span>Latency: {resultInfo.latency_ms?.toFixed(1)}ms</span>
                    </>
                  )}
                  {resultInfo.type === "search" && (
                    <>
                      <span>Query: "{resultInfo.query}"</span>
                      <span>Count: {resultInfo.count}</span>
                      <span>Method: {resultInfo.method}</span>
                      <span>Latency: {resultInfo.latency_ms?.toFixed(1)}ms</span>
                    </>
                  )}
                  {resultInfo.type === "similar" && (
                    <>
                      <span>Product: {resultInfo.product_id}</span>
                      <span>Count: {resultInfo.count}</span>
                      <span>Method: {resultInfo.method}</span>
                    </>
                  )}
                </div>
              </div>
            )}

            {/* Results Display */}
            <div className={styles.resultsSection}>
              {demoTab === "recommend" && recommendations.length > 0 && (
                <div className={styles.productsGrid}>
                  {recommendations.map((product, idx) => (
                    <ProductCard
                      key={`rec-${idx}`}
                      product={product}
                      addProductToCart={addProductToCart}
                    />
                  ))}
                </div>
              )}

              {demoTab === "search" && searchResults.length > 0 && (
                <div className={styles.productsGrid}>
                  {searchResults.map((product, idx) => (
                    <ProductCard
                      key={`search-${idx}`}
                      product={product}
                      addProductToCart={addProductToCart}
                    />
                  ))}
                </div>
              )}

              {demoTab === "similar" && similarItems.length > 0 && (
                <div className={styles.productsGrid}>
                  {similarItems.map((product, idx) => (
                    <ProductCard
                      key={`similar-${idx}`}
                      product={product}
                      addProductToCart={addProductToCart}
                    />
                  ))}
                </div>
              )}
            </div>
          </div>
        )}

        {/* Streamlit Dashboard Tab */}
        {activeTab === "dashboard" && (
          <div className={styles.dashboardSection}>
            <div className={styles.dashboardHeader}>
              <h3>📊 VieComRec Monitoring Dashboard</h3>
              <p>Full dashboard với Service Health, Training History, Scheduler Management, Drift Detection</p>
              <a
                href={STREAMLIT_URL}
                target="_blank"
                rel="noopener noreferrer"
                className={styles.openNewTab}
              >
                🔗 Mở trong tab mới
              </a>
            </div>
            
            <div className={styles.iframeContainer}>
              {!streamlitLoaded && (
                <div className={styles.iframeLoading}>
                  <div className={styles.spinner}></div>
                  <p>Đang tải Streamlit Dashboard...</p>
                  <small>Đảm bảo đã chạy: <code>streamlit run service/dashboard.py</code></small>
                </div>
              )}
              <iframe
                src={STREAMLIT_URL}
                title="VieComRec Dashboard"
                className={styles.dashboardIframe}
                onLoad={() => setStreamlitLoaded(true)}
                onError={() => setStreamlitLoaded(false)}
              />
            </div>
          </div>
        )}

        {/* API Docs Tab */}
        {activeTab === "docs" && (
          <div className={styles.docsSection}>
            <div className={styles.docsHeader}>
              <h3>📚 VieComRec API Documentation</h3>
              <p>OpenAPI/Swagger documentation</p>
              <a
                href={`${VIECOMREC_URL}/docs`}
                target="_blank"
                rel="noopener noreferrer"
                className={styles.openNewTab}
              >
                🔗 Mở Swagger UI
              </a>
            </div>
            
            <div className={styles.iframeContainer}>
              <iframe
                src={`${VIECOMREC_URL}/docs`}
                title="API Documentation"
                className={styles.docsIframe}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default AIDashboard;
