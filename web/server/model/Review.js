import mongoose from "mongoose";

const reviewSchema = new mongoose.Schema({
    review_id: {
        type: String,
        unique: true,
        required: true
    },
    user_id: {
        type: String,  // Changed to String to accept various ID formats
        required: false
    },
    product_id: {
        type: Number,
        required: true
    },
    order_id: {
        type: String,
        ref: 'Order'
    },
    rating: {
        type: Number,
        required: true,
        min: 1,
        max: 5
    },
    comment: {
        type: String,
        default: ""
    },
    images: [{
        type: String
    }],
    helpful_count: {
        type: Number,
        default: 0
    },
    status: {
        type: String,
        enum: ['PENDING', 'APPROVED', 'REJECTED'],
        default: 'PENDING'
    },
    created_at: {
        type: Date,
        default: Date.now
    },
    updated_at: {
        type: Date,
        default: Date.now
    }
});

const Review = mongoose.model('Review', reviewSchema);
export default Review;
