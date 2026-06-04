import Order from "../../model/Orders.js";
import Pagination from "../../utils/pagination.js";
import axios from "axios";
import {
  USER_BASEURL,
  PRODUCTS_BASEURL,
  NOTIFICATIONS_BASEURL,
  SHIPPING_BASEURL,
} from "../../services/BaseURLs.js";

export const createOrder = async (req, res) => {
  try {
    const { data } = req.body;
    console.log("📦 Creating order with data:", data.order_id);
    
    const order = new Order({
      order_id: data.order_id,
      user_id: data.user_id || null,
      name: {
        first: data.firstName,
        last: data.lastName,
      },
      email: data.email,
      phone_number: data.phone_number,
      address: JSON.parse(data.address),
      ordered_at: Date.now(),
      products: JSON.parse(data.products),
      total: data.total,
      status: 'PROCESSING'
    });

    await order.save();
    console.log("✅ Order saved to database");

    // Update product quantities (non-blocking, don't fail if this fails)
    try {
      await axios.patch(`${PRODUCTS_BASEURL}/updateQuantity`, {
        products: order.products,
      });
      console.log("✅ Product quantities updated");
    } catch (err) {
      console.warn("⚠️ Failed to update product quantities:", err.message);
    }

    // Send notification (non-blocking, don't fail if this fails)
    try {
      const to = order.email;
      await axios.post(`${NOTIFICATIONS_BASEURL}/order-confirmation`, {
        to,
        order,
      });
      console.log("✅ Notification sent");
    } catch (err) {
      console.warn("⚠️ Failed to send notification:", err.message);
    }

    // Create shipment record
    try {
      await axios.post(SHIPPING_BASEURL, {
        order_id: order.order_id,
        ordered_at: order.ordered_at,
        address: order.address,
        total: order.total,
      });
      console.log("✅ Shipment created");
    } catch (err) {
      console.warn("⚠️ Failed to create shipment:", err.message);
    }

    res.status(200).json({ order_id: order.order_id });
  } catch (error) {
    console.error("❌ Order creation error:", error);
    res.status(500).json({ error: error.message });
  }
};

export const getOrder = async (req, res) => {
  try {
    const requiredOrder = await Order.findOne({ order_id: req.params.id });

    if (!requiredOrder) {
      return res.status(404).json({ message: "Order does not exist" });
    }

    const productIds = requiredOrder.products.map((pr) => pr.product_id);
    const { data } = await axios.post(`${PRODUCTS_BASEURL}/arr`, {
      arr: productIds,
    });

    res.status(200).json({ ...requiredOrder._doc, products: data });
  } catch (error) {
    res.status(400).json({ message: error.message });
  }
};

export const getAllOrders = async (req, res) => {
  try {
    const id = req.body.id;

    // verify the user's role by calling the `User` service
    try {
      await axios.post(`${USER_BASEURL}/role`, { id, role: "ADMIN" });
    } catch (e) {
      const { response } = e;
      return res.status(response.status).json(response.data);
    }

    const orders = await Order.find().sort({ ordered_at: -1 });
    const ordersPaged = Pagination(req.query.page, orders);

    const total_pages = Math.ceil((await Order.count()) / 20);

    res.status(200).json({ total_pages, orders: ordersPaged });
  } catch (error) {
    res.status(404).json({ message: error.message });
  }
};

export const updateOrder = async (req, res) => {
  try {
    const orderStatus = req.body.status;
    const id = req.body.id;

    // verify the user's role by calling the `User` service
    try {
      await axios.post(`${USER_BASEURL}/role`, { id, role: "ADMIN" });
    } catch (e) {
      const { response } = e;
      return res.status(response.status).json(response.data);
    }

    if (
      !["CREATED", "PROCESSING", "FULFILLED", "CANCELLED"].includes(orderStatus)
    ) {
      return res
        .status(400)
        .json({
          message:
            "Invalid status, has to be CREATED, PROCESSING, FULFILLED, CANCELLED",
        });
    }

    const updatedOrder = await Order.findOneAndUpdate(
      { order_id: req.params.id },
      {
        status: orderStatus,
      }
    );

    if (!updatedOrder) {
      return res.status(404).json({ message: "Order does not exist" });
    }

    // Update shipment status based on order status
    try {
      let shipmentStatus = "PROCESSING";
      if (orderStatus === "FULFILLED") {
        shipmentStatus = "SHIPPED";
      } else if (orderStatus === "CANCELLED") {
        shipmentStatus = "CANCELLED";
      } else if (orderStatus === "PROCESSING") {
        shipmentStatus = "PROCESSING";
      }
      
      // Use internal route (no auth required)
      await axios.patch(`${SHIPPING_BASEURL}/internal/${req.params.id}`, {
        status: shipmentStatus,
      });
      console.log(`✅ Shipment status updated to ${shipmentStatus}`);

      // Auto-deliver after 5 seconds (demo mode) and request review
      if (orderStatus === "FULFILLED") {
        console.log(`🚚 Simulating delivery for order ${req.params.id}...`);
        
        setTimeout(async () => {
          try {
            // Update shipment to DELIVERED (use internal route)
            await axios.patch(`${SHIPPING_BASEURL}/internal/${req.params.id}`, {
              status: "DELIVERED",
            });
            console.log(`✅ Order ${req.params.id} has been DELIVERED!`);

            // Update order status to DELIVERED
            await Order.findOneAndUpdate(
              { order_id: req.params.id },
              { status: "FULFILLED", delivered_at: new Date() }
            );

            // Send review request notification
            try {
              await axios.post(`${NOTIFICATIONS_BASEURL}/review-request`, {
                to: updatedOrder.email,
                order_id: req.params.id,
                products: updatedOrder.products,
                customer_name: `${updatedOrder.name.first} ${updatedOrder.name.last}`
              });
              console.log(`📧 Review request sent to ${updatedOrder.email}`);
            } catch (notifErr) {
              console.warn("⚠️ Failed to send review request:", notifErr.message);
            }
          } catch (deliveryErr) {
            console.error("❌ Auto-delivery failed:", deliveryErr.message);
          }
        }, 5000); // 5 seconds delay
      }
    } catch (err) {
      console.warn("⚠️ Failed to update shipment:", err.message);
    }

    res.status(200).json(updatedOrder);
  } catch (error) {
    res.status(404).json({ message: error.message });
  }
};
