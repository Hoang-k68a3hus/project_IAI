/**
 * Script Import Data từ VieComRec vào MongoDB
 * 
 * Chạy: cd server && node scripts/importViecomrecData.js
 */

import mongoose from "mongoose";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import { parse } from "csv-parse/sync";
import dotenv from "dotenv";

dotenv.config();
mongoose.set("strictQuery", true);

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Data paths
const DATA_DIR =
    process.env.VIECOMREC_DATA_DIR ||
    path.resolve(__dirname, "..", "..", "..", "data", "published_data");
const PRODUCTS_CSV = path.join(DATA_DIR, "data_product.csv");
const ATTRIBUTES_CSV = path.join(DATA_DIR, "attribute_based_embeddings/attribute_text_filtering_merged.csv");
const REVIEWS_CSV = path.join(DATA_DIR, "data_reviews_purchase.csv");

// MongoDB connection
const MONGO_URI = process.env.CONNECTION_URL || "mongodb://admin:password123@localhost:27017/cosmetic_db?authSource=admin";

// Import models
import Products from "../model/Products.js";
import Reviews from "../model/Reviews.js";

// Parse CSV file
function parseCSV(filePath) {
    console.log(`📖 Reading: ${filePath}`);
    const content = fs.readFileSync(filePath, "utf-8");
    const records = parse(content, {
        columns: true,
        skip_empty_lines: true,
        relax_column_count: true,
        trim: true
    });
    console.log(`   ✅ Loaded ${records.length} records`);
    return records;
}

// Import Products
async function importProducts() {
    console.log("\n🛍️  IMPORTING PRODUCTS...");
    
    // Load products data
    const products = parseCSV(PRODUCTS_CSV);
    
    // Load attributes data
    let attributesMap = {};
    try {
        const attributes = parseCSV(ATTRIBUTES_CSV);
        attributes.forEach(attr => {
            attributesMap[attr.product_id] = attr;
        });
        console.log(`   📋 Loaded ${Object.keys(attributesMap).length} product attributes`);
    } catch (e) {
        console.log("   ⚠️  No attributes file found, skipping...");
    }
    
    let imported = 0;
    let errors = 0;
    
    for (const row of products) {
        try {
            const productId = parseInt(row.product_id);
            if (isNaN(productId)) continue;
            
            // Get attributes if available
            const attr = attributesMap[row.product_id] || {};
            
            const productData = {
                product_id: productId,
                shop_id: parseInt(row.shop_id) || 0,
                name: row.product_name,
                product_name: row.product_name,
                brand: row.brand || attr.brand || "Unknown",
                price: parseFloat(row.price) || 0,
                
                // Rating & Sales
                avg_rating: parseFloat(row.avg_star) || 0,
                avg_star: parseFloat(row.avg_star) || 0,
                num_sold: parseInt(row.num_sold_time) || 0,
                num_sold_time: parseInt(row.num_sold_time) || 0,
                num_rating: parseInt(row.num_rating) || 0,
                
                // Rating breakdown
                is_5_star: parseInt(row.is_5_star) || 0,
                is_4_star: parseInt(row.is_4_star) || 0,
                is_3_star: parseInt(row.is_3_star) || 0,
                is_2_star: parseInt(row.is_2_star) || 0,
                is_1_star: parseInt(row.is_1_star) || 0,
                is_commented: parseInt(row.is_commented) || 0,
                is_image: parseInt(row.is_image) || 0,
                
                // Product details
                category: row.type || attr.type || "cosmetic",
                type: row.type || attr.type || "no_type",
                skin_kind: row.skin_kind || attr.skin_kind || "no_skin",
                skin_type: attr.skin_type || "",
                origin: row.origin || attr.origin || "no_origin",
                expiry: row.expiry || attr.expiry || "no_expiry",
                send_from: row.send_from || "",
                storage: parseFloat(row.storage) || 0,
                variation: row.variation || "",
                capacity: attr.capacity || "",
                design: attr.design || "",
                
                // Content
                description: row.processed_description || "",
                processed_description: row.processed_description || "",
                ingredient: attr.ingredient || "",
                feature: attr.feature || "",
                
                // Image - use product_id to construct image path
                image: `/images/products/${row.image_path || productId + '.jpg'}`,
                image_path: row.image_path || "",
                
                // Stock
                stock: 100
            };
            
            await Products.updateOne(
                { product_id: productId },
                { $set: productData },
                { upsert: true }
            );
            
            imported++;
            if (imported % 500 === 0) {
                console.log(`   📦 Imported ${imported} products...`);
            }
        } catch (e) {
            errors++;
            if (errors <= 5) {
                console.error(`   ❌ Error importing product ${row.product_id}:`, e.message);
            }
        }
    }
    
    console.log(`   ✅ Products imported: ${imported}, Errors: ${errors}`);
    return imported;
}

// Import Reviews
async function importReviews() {
    console.log("\n💬 IMPORTING REVIEWS...");
    
    const reviews = parseCSV(REVIEWS_CSV);
    
    let imported = 0;
    let errors = 0;
    
    // Process in batches
    const batchSize = 1000;
    const batches = [];
    
    for (let i = 0; i < reviews.length; i += batchSize) {
        batches.push(reviews.slice(i, i + batchSize));
    }
    
    console.log(`   📊 Processing ${reviews.length} reviews in ${batches.length} batches...`);
    
    for (const batch of batches) {
        const operations = [];
        
        for (const row of batch) {
            try {
                const reviewId = parseInt(row.review_id || row["Unnamed: 0"]);
                const userId = parseInt(row.user_id);
                const productId = parseInt(row.product_id);
                const rating = parseInt(row.rating);
                
                if (isNaN(reviewId) || isNaN(userId) || isNaN(productId) || isNaN(rating)) continue;
                
                operations.push({
                    updateOne: {
                        filter: { review_id: reviewId },
                        update: {
                            $set: {
                                review_id: reviewId,
                                user_id: userId,
                                product_id: productId,
                                shop_id: parseInt(row.shop_id) || 0,
                                rating: rating,
                                product_quality: parseFloat(row.product_quality) || rating,
                                comment: row.processed_comment || "",
                                processed_comment: row.processed_comment || "",
                                product_name: row.product_name_x || row.product_name || "",
                                variation: row.variation_x || row.variation || "",
                                cmt_date: row.cmt_date ? new Date(row.cmt_date) : new Date()
                            }
                        },
                        upsert: true
                    }
                });
            } catch (e) {
                errors++;
            }
        }
        
        if (operations.length > 0) {
            await Reviews.bulkWrite(operations, { ordered: false });
            imported += operations.length;
            console.log(`   💬 Imported ${imported} reviews...`);
        }
    }
    
    console.log(`   ✅ Reviews imported: ${imported}, Errors: ${errors}`);
    return imported;
}

// Main function
async function main() {
    console.log("🚀 VieComRec Data Import Script");
    console.log("================================\n");
    console.log(`📁 Data directory: ${DATA_DIR}`);
    console.log(`🔗 MongoDB: ${MONGO_URI}\n`);
    
    try {
        // Connect to MongoDB
        console.log("🔌 Connecting to MongoDB...");
        await mongoose.connect(MONGO_URI);
        console.log("✅ MongoDB connected!\n");
        
        // Import data
        const productsCount = await importProducts();
        const reviewsCount = await importReviews();
        
        // Summary
        console.log("\n================================");
        console.log("🎉 IMPORT COMPLETED!");
        console.log(`   📦 Products: ${productsCount}`);
        console.log(`   💬 Reviews: ${reviewsCount}`);
        console.log("================================\n");
        
    } catch (error) {
        console.error("❌ Error:", error);
    } finally {
        await mongoose.disconnect();
        console.log("🔌 MongoDB disconnected");
    }
}

main();
