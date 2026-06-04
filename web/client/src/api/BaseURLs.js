const trimTrailingSlash = (value) => value.replace(/\/+$/, "");

export const API_BASEURL = trimTrailingSlash(
  process.env.REACT_APP_API_URL || "http://localhost:5000/api"
);
export const WEB_SERVER_BASEURL = API_BASEURL.replace(/\/api$/, "");

// VieComRec AI Recommendation API
export const VIECOMREC_BASEURL = trimTrailingSlash(
  process.env.REACT_APP_VIECOMREC_API_URL || "http://localhost:8000"
);
export const STREAMLIT_BASEURL = trimTrailingSlash(
  process.env.REACT_APP_STREAMLIT_URL || "http://localhost:8501"
);

// Backend API URLs
export const PRODUCTS_BASEURL = `${API_BASEURL}/products`;
export const INGEST_BASEURL = `${API_BASEURL}/ingest`;
export const ORDERS_BASEURL = `${API_BASEURL}/orders`;
export const SHIPPING_BASEURL = `${API_BASEURL}/shipping`;
export const PAYMENTS_BASEURL = `${API_BASEURL}/payments`;
export const USER_BASEURL = `${API_BASEURL}/auth`;

export const PRODUCT_IMAGE_BASEURL = trimTrailingSlash(
  process.env.REACT_APP_PRODUCT_IMAGE_BASE_URL ||
    `${WEB_SERVER_BASEURL}/images/products`
);
