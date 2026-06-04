import mongoose from "mongoose";
import fs from "fs";
import path from "path";
import csv from "csv-parser";
import Products from "../model/Products.js";
import dotenv from "dotenv";

dotenv.config();
mongoose.set("strictQuery", true);

// ⚠️ SỬA CHO ĐÚNG CHUỖI KẾT NỐI CỦA BẠN
const MONGO_URI =
  process.env.CONNECTION_URL ||
  process.env.MONGO_URL ||
  "mongodb://admin:password123@localhost:27017/cosmetic_db?authSource=admin";

const CSV_FILE =
  process.env.PRODUCTS_CSV_FILE || path.join(process.cwd(), "data_product.csv");

async function main() {
  await mongoose.connect(MONGO_URI);
  console.log("🔥 Connected to MongoDB");

  const rows = [];

  fs.createReadStream(CSV_FILE)
    .pipe(csv())
    .on("data", (row) => rows.push(row))
    .on("end", async () => {
      console.log(`📄 Loaded ${rows.length} rows from CSV`);

      for (const row of rows) {
        const productId = row.product_id?.toString().trim();
        let imgPath = row.image_path?.toString().trim();
        const name = row.product_name?.toString().trim();
        const price = Number(row.price) || 0;

        if (!productId) continue;

        if (imgPath && imgPath.match(/\.mp4$/i)) {
          imgPath = imgPath.replace(/\.mp4$/i, ".jpg");
        } else if (imgPath && !imgPath.match(/\.(jpg|jpeg|png|webp)$/i)) {
          imgPath += ".jpg";
        }

        const image = `/images/products/${imgPath}`;

        await Products.updateOne(
          { product_id: productId },
          {
            $set: {
              name,
              price,
              image,
            },
          },
          { upsert: true }
        );
      }

      console.log("🎉 DONE: Products updated successfully!");
      await mongoose.disconnect();
      console.log("🔌 MongoDB disconnected");
    });
}

main();
