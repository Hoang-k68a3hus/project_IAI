import mongoose from "mongoose";
const { Schema } = mongoose;

const reviewSchema = new Schema({
    review_id: { type: Number },
    user_id: { type: Number, required: true, index: true },
    product_id: { type: Number, required: true, index: true },
    shop_id: { type: Number, default: 0 },
    rating: { type: Number, required: true, min: 1, max: 5 },
    product_quality: { type: Number, min: 1, max: 5 },
    
    // Review content
    comment: String,
    processed_comment: String,
    
    // Product info at time of review
    product_name: String,
    variation: String,
    
    // Metadata
    cmt_date: { type: Date },
    created_at: { type: Date, default: Date.now }
}, {
    timestamps: true
});

// Compound index for user-product pair
reviewSchema.index({ user_id: 1, product_id: 1 });
reviewSchema.index({ product_id: 1, rating: -1 });

const Reviews = mongoose.model('Reviews', reviewSchema);
export default Reviews;
