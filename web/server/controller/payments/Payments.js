import axios from "axios";
import jwt from "jsonwebtoken";
import generateId from "../../utils/generateId.js";
import {ORDERS_BASEURL, WEBSITE_BASE_URL} from "../../services/BaseURLs.js";

const VIECOMREC_API = process.env.VIECOMREC_API || "http://localhost:8000";

// Check if Stripe is configured
const isStripeConfigured = process.env.STRIPE_SECRET_KEY && 
    process.env.STRIPE_SECRET_KEY !== 'your_stripe_secret_key';

let stripe = null;
if (isStripeConfigured) {
    const Stripe = (await import("stripe")).default;
    stripe = new Stripe(process.env.STRIPE_SECRET_KEY);
}

export const createCheckoutSession = async (req, res) => {
    try {
        const {products, total} = jwt.verify(
            req.body.token,
            process.env.JWT_SECRET_KEY
        );
        const {data} = req.body;
        const order_id = generateId();

        // Demo mode: Skip Stripe if not configured
        if (!isStripeConfigured) {
            console.log("⚠️ Stripe not configured - using DEMO MODE");
            console.log("📦 Creating order:", order_id);
            
            // Create order directly without payment
            const orderData = {
                order_id: order_id,
                firstName: data.name.first,
                lastName: data.name.last,
                email: data.email,
                phone_number: data.phone_number,
                address: JSON.stringify({
                    country: data.address.country,
                    city: data.address.city,
                    area: data.address.area,
                    street: data.address.street,
                    building_number: data.address.building_number,
                    floor: data.address.floor,
                    apartment_number: data.address.apartment_number,
                }),
                products: JSON.stringify(
                    products.map((product) => ({
                        product_id: product.product_id,
                        name: product.name,
                        quantity: product.quantity,
                    }))
                ),
                total: total,
            };

            // Create order in database
            try {
                console.log("📤 Sending order to:", ORDERS_BASEURL);
                await axios.post(ORDERS_BASEURL, { data: orderData });
                console.log("✅ Order created successfully");
            } catch (orderErr) {
                console.error("❌ Order creation failed:", orderErr.response?.data || orderErr.message);
                throw orderErr;
            }

            // Send purchase events to VieComRec
            try {
                for (const product of products) {
                    try {
                        await axios.post(
                            `${VIECOMREC_API}/ingest/purchase`,
                            {
                                user_id: 1,
                                product_id: parseInt(product.product_id),
                                quantity: parseInt(product.quantity) || 1,
                                timestamp: new Date().toISOString()
                            },
                            { timeout: 5000 }
                        );
                    } catch (err) {
                        console.warn(`Failed to send purchase event:`, err.message);
                    }
                }
            } catch (err) {
                console.warn("VieComRec error:", err.message);
            }

            // Redirect to success page directly
            return res.status(201).json({ 
                url: `${WEBSITE_BASE_URL}/checkout/success?order=${order_id}`,
                demo: true 
            });
        }

        // Production mode: Use Stripe
        const session = await stripe.checkout.sessions.create({
            payment_method_types: ["card"],
            mode: "payment",
            line_items: products.map((product) => {
                return {
                    price_data: {
                        currency: "vnd",
                        product_data: {
                            name: product.name,
                        },
                        // VND is zero-decimal currency, no need to multiply by 100
                        unit_amount: parseInt(product.price),
                    },
                    quantity: product.quantity || 1,
                };
            }),
            payment_intent_data: {
                metadata: {
                    order_id: order_id,
                    firstName: data.name.first,
                    lastName: data.name.last,
                    email: data.email,
                    phone_number: data.phone_number,
                    address: JSON.stringify({
                        country: data.address.country,
                        city: data.address.city,
                        area: data.address.area,
                        street: data.address.street,
                        building_number: data.address.building_number,
                        floor: data.address.floor,
                        apartment_number: data.address.apartment_number,
                    }),
                    products: JSON.stringify(
                        products.map((product) => {
                            return {
                                product_id: product.product_id,
                                name: product.name,
                                quantity: product.quantity,
                            };
                        })
                    ),
                    total: total,
                },
            },
            success_url: `${WEBSITE_BASE_URL}/checkout/success?order=${order_id}`,
            cancel_url: `${WEBSITE_BASE_URL}/cart`,
        });

        res.status(201).json({url: session.url});
    } catch (error) {
        res.status(500).json({message: error.message});
    }
};

export const webhook = async (req, res) => {
    const eventType = req.body.type;
    const {metadata} = req.body.data.object;
    try {
        if (eventType === "charge.succeeded") {
            // Create order in MongoDB
            await axios.post(ORDERS_BASEURL, {data: metadata});

            // Parse products and send to VieComRec
            try {
                const products = JSON.parse(metadata.products);
                // Simulate user_id from email (in production, this should come from authenticated session)
                const user_id = parseInt(metadata.email.split('@')[0]) || 1;

                // Send each purchase to VieComRec
                for (const product of products) {
                    try {
                        await axios.post(
                            `${VIECOMREC_API}/ingest/purchase`,
                            {
                                user_id,
                                product_id: parseInt(product.product_id),
                                quantity: parseInt(product.quantity) || 1,
                                timestamp: new Date().toISOString()
                            },
                            { timeout: 5000 }
                        );
                    } catch (err) {
                        console.warn(`Failed to send purchase event for product ${product.product_id}:`, err.message);
                        // Continue with other products even if one fails
                    }
                }
            } catch (err) {
                console.warn("Failed to parse products for VieComRec:", err.message);
                // Still return success since order was created
            }
        }
        res.status(200).json(metadata);
    } catch (error) {
        res.status(404).json({message: error.message});
    }
};

