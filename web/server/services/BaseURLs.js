// VieComRec AI Recommendation API
export const VIECOMREC_BASEURL = process.env.VIECOMREC_API || 'http://localhost:8000';

// Local development URLs (monolithic server on port 5000)
export const WEBSITE_BASE_URL = process.env.WEBSITE_URL || 'http://localhost:3000';
export const SERVER_BASE_URL = process.env.SERVER_URL || 'http://localhost:5000';
export const PRODUCTS_BASEURL = process.env.PRODUCTS_URL || 'http://localhost:5000/api/products';
export const ORDERS_BASEURL = process.env.ORDERS_URL || 'http://localhost:5000/api/orders';
export const SHIPPING_BASEURL = process.env.SHIPPING_URL || 'http://localhost:5000/api/shipping';
export const NOTIFICATIONS_BASEURL = process.env.NOTIFICATIONS_URL || 'http://localhost:5000/api/notifications';
export const PAYMENTS_BASEURL = process.env.PAYMENTS_URL || 'http://localhost:5000/api/payments';
export const USER_BASEURL = process.env.USERS_URL || 'http://localhost:5000/api/auth';
