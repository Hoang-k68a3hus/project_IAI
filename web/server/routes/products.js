import express from "express";

import {
  PostProducts,
  ShowProductsPerPage,
  productsSearch,
  validateCart,
  adminUpdateProducts,
  ProductsRecommendations,
  getProductsArr,
  updateQuantity,
  getProductById,
  getProductReviews,
  getSimilarProducts
} from "../controller/products/Products.js";
import auth from "../middleware/auth.js";

const router = express.Router();

router.get("/", ShowProductsPerPage);
router.get("/search", productsSearch);
router.get("/recommendations", ProductsRecommendations);
router.get("/:id", getProductById);
router.get("/:id/reviews", getProductReviews);
router.get("/:id/similar", getSimilarProducts);
router.post("/", auth, PostProducts);
router.patch("/", auth, adminUpdateProducts);
router.post("/cart", validateCart);
router.post("/arr", getProductsArr);
router.patch("/updateQuantity", updateQuantity);

export default router;
