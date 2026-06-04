import express from "express";
import dotenv from "dotenv";
import cors from "cors";
import path from "path";
import { fileURLToPath } from "url";
import mongoose from "mongoose";
import recommend from "./routes/recommend.js";

// 👉 __dirname cho ES modules
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

dotenv.config();
mongoose.set("strictQuery", true);

const app = express();
app.use(cors());
app.use(express.json());

// 👉 Static images - VieComRec data folder
const VIECOMREC_IMAGES = process.env.VIECOMREC_IMAGE_DIR
  ? path.resolve(process.env.VIECOMREC_IMAGE_DIR)
  : path.resolve(__dirname, "..", "..", "data", "published_data", "image");
app.use("/images/products", express.static(VIECOMREC_IMAGES));
console.log(`📸 Serving images from: ${VIECOMREC_IMAGES}`);

// 👉 Fallback static images from local public folder
app.use("/images", express.static(path.join(__dirname, "public", "images")));

import authRoutes from "./routes/auth.js";
import productRoutes from "./routes/products.js";
import ingestRoutes from "./routes/ingest.js";
import ordersRoutes from "./routes/orders.js";
import paymentsRoutes from "./routes/payments.js";
import shippingRoutes from "./routes/shipping.js";
import notificationsRoutes from "./routes/notifications.js";

app.use("/api/auth", authRoutes);
app.use("/api/products", productRoutes);
app.use("/api/ingest", ingestRoutes);
app.use("/api/recommend", recommend);
app.use("/api/orders", ordersRoutes);
app.use("/api/payments", paymentsRoutes);
app.use("/api/shipping", shippingRoutes);
app.use("/api/notifications", notificationsRoutes);
app.get("/api/health", (req, res) => {
  res.json({
    status: "ok",
    mongo: mongoose.connection.readyState,
    timestamp: new Date().toISOString(),
  });
});

// 👉 KẾT NỐI MONGODB TẠI ĐÂY
const connectionUrl =
  process.env.CONNECTION_URL ||
  process.env.MONGO_URL ||
  "mongodb://admin:password123@localhost:27017/cosmetic_db?authSource=admin";

mongoose
  .connect(connectionUrl, {
    useNewUrlParser: true,
    useUnifiedTopology: true,
  })
  .then(() => console.log("✅ MongoDB Connected"))
  .catch((err) => console.log("❌ MongoDB Error:", err));

// 👉 Start server
app.listen(process.env.PORT || 5000, () => {
  console.log("Server running...");
});
