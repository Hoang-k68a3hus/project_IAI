import express from "express";
import {orderConfirmation, reviewRequest} from "../controller/notifications/Notifications.js";

const router = express.Router();

router.post('/order-confirmation', orderConfirmation);
router.post('/review-request', reviewRequest);

export default router;
