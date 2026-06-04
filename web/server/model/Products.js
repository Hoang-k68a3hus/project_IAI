import mongoose from "mongoose";
const {Schema} = mongoose;

const productSchema = new Schema({
    product_id: {type: Number, unique: true, required: true},
    shop_id: {type: Number, default: 0},
    name: String,
    product_name: String,
    brand: String,
    price: {type: Number, default: 0},
    
    // Rating & Sales
    avg_rating: {type: Number, default: 0},
    avg_star: {type: Number, default: 0},
    num_sold: {type: Number, default: 0},
    num_sold_time: {type: Number, default: 0},
    num_rating: {type: Number, default: 0},
    
    // Rating breakdown
    is_5_star: {type: Number, default: 0},
    is_4_star: {type: Number, default: 0},
    is_3_star: {type: Number, default: 0},
    is_2_star: {type: Number, default: 0},
    is_1_star: {type: Number, default: 0},
    is_commented: {type: Number, default: 0},
    is_image: {type: Number, default: 0},
    
    // Product details
    category: String,
    type: String,
    skin_kind: String,
    skin_type: String,
    origin: String,
    expiry: String,
    send_from: String,
    storage: Number,
    variation: String,
    capacity: String,
    design: String,
    
    // Content
    description: String,
    processed_description: String,
    ingredient: String,
    feature: String,
    
    // Image
    image: String,
    image_path: String,
    
    // Stock
    stock: {
        type: Number,
        default: 100
    }
}, {
    timestamps: true
});

// Index for search
productSchema.index({ name: 'text', product_name: 'text', brand: 'text', description: 'text' });

const Products = mongoose.model('Products', productSchema);
export default Products;
