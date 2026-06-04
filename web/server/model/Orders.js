import mongoose from "mongoose";

const orderSchema = mongoose.Schema({
    order_id: {
        type: String,
        required: true,
        unique: true
    },
    user_id: {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'Users'
    },
    name: {
        first: String,
        last: String
    },
    email: {type: String, required: false},
    phone_number: {type: String, required: false},
    address: {
        country: String,
        city: String,
        area: String,
        street: String,
        building_number: String,
        floor: String,
        apartment_number: String
    },
    ordered_at: {
        type: Date,
        default: Date.now
    },
    status: {
        type: String,
        enum: ['CREATED', 'PROCESSING', 'FULFILLED', 'CANCELLED'],
        default: 'CREATED'
    },
    products: {
        type: Array,
        required: true,
        default: []
    },
    total: {
        type: Number,
        required: true
    },
    // Tracking data sent to ML
    ingested: {
        type: Boolean,
        default: false
    },
    ingest_timestamp: {
        type: Date
    }
});

const Order = mongoose.model('Order', orderSchema);

export default Order;