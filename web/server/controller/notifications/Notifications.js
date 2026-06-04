import sgMail from '@sendgrid/mail';
import OrderConfirmationTemplate from "./templates/OrderConfirmationTemplate.js";

const defaults = {
    from: {
        name: 'Cosmetic Shop',
        email: 'hey.baraa@gmail.com',
    }
}

export const orderConfirmation = async (req, res) => {
    const {to, order} = req.body;
    
    const email = {
        ...defaults,
        to,
        subject: "Xác nhận đơn hàng",
        text: `Đơn hàng #${order.order_id} đã được đặt thành công. ${order.products.length} sản phẩm sẽ được giao đến ${order.address}. Tổng tiền: ${order.total} VNĐ`,
        html: OrderConfirmationTemplate(order)
    }

    await sendEmail(email, res);
}

export const reviewRequest = async (req, res) => {
    const {to, order_id, products, customer_name} = req.body;
    
    const productNames = products.map(p => p.name).join(', ');
    
    const email = {
        ...defaults,
        to,
        subject: `Đánh giá đơn hàng #${order_id}`,
        text: `Xin chào ${customer_name}! Đơn hàng #${order_id} của bạn đã được giao thành công. Hãy đánh giá sản phẩm để giúp chúng tôi cải thiện dịch vụ!`,
        html: ReviewRequestTemplate(order_id, products, customer_name)
    }

    await sendEmail(email, res);
}

const ReviewRequestTemplate = (order_id, products, customer_name) => {
    const productList = products.map(p => `<li>${p.name}</li>`).join('');
    return `
        <div style="font-family: Arial, sans-serif; max-width: 600px; margin: 0 auto;">
            <h2 style="color: #e91e63;">🎉 Đơn hàng đã giao thành công!</h2>
            <p>Xin chào <strong>${customer_name}</strong>,</p>
            <p>Đơn hàng <strong>#${order_id}</strong> của bạn đã được giao thành công!</p>
            <h3>Sản phẩm đã mua:</h3>
            <ul>${productList}</ul>
            <div style="background: #fff3e0; padding: 20px; border-radius: 10px; margin: 20px 0;">
                <h3 style="color: #ff9800;">⭐ Hãy đánh giá sản phẩm!</h3>
                <p>Ý kiến của bạn rất quan trọng với chúng tôi. Hãy dành chút thời gian để đánh giá sản phẩm bạn đã mua.</p>
                <a href="http://localhost:3000/orders/${order_id}" 
                   style="display: inline-block; background: #e91e63; color: white; padding: 12px 24px; text-decoration: none; border-radius: 5px; margin-top: 10px;">
                    Đánh giá ngay
                </a>
            </div>
            <p>Cảm ơn bạn đã mua sắm tại Cosmetic Shop! 💖</p>
        </div>
    `;
}

const sendEmail = async (email, res) => {
    try {
        // Check if SendGrid is configured
        if (!process.env.SENDGRID_API_KEY || process.env.SENDGRID_API_KEY === 'your_sendgrid_api_key') {
            console.log("📧 [DEMO] Email would be sent to:", email.to);
            console.log("📧 [DEMO] Subject:", email.subject);
            return res.status(200).json({email, result: 'Demo mode - Email logged'});
        }
        
        await sgMail.send(email);
        res.status(200).json({email, result: 'Sent Successfully'});
    } catch (e) {
        console.warn("⚠️ Email send failed:", e.message);
        res.status(400).json({message: e.message});
    }
}