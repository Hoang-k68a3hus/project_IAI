import express from "express";
import {postShipments, updateShipments, updateShipmentsInternal, getShipmentId, getShipments} from "../controller/shipping/Shipping.js";
import auth from "../middleware/auth.js";


const router = express.Router();
router.post('/', postShipments);
router.get('/:id', getShipmentId);
router.get('/', auth, getShipments);
router.patch('/:id', auth, updateShipments);
// Internal route for server-to-server calls (no auth required)
router.patch('/internal/:id', updateShipmentsInternal);


export default router;