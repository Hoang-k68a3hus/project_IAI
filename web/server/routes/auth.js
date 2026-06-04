import express from "express";
import auth from "../middleware/auth.js";
import {
  login,
  register,
  verifyRole,
  verifyUser,
} from "../controller/me/Authentication.js";
import { getWishlist, updateWishlist } from "../controller/me/Wishlist.js";

const router = express.Router();

router.post("/register", register);
router.post("/login", login);
router.post("/verify", auth, verifyUser);
router.post("/role", verifyRole);

// MUST BE POST
router.post("/wishlist", auth, getWishlist);
router.patch("/wishlist", auth, updateWishlist);

export default router;
