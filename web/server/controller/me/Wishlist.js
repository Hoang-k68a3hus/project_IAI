import Users from "../../model/Users.js";
import axios from "axios";
import { PRODUCTS_BASEURL } from "../../services/BaseURLs.js";

export const getWishlist = async (req, res) => {
  const id = req.body.id;

  try {
    const user = await Users.findById(id);
    if (!user) {
      return res.status(404).json({ message: "User not found" });
    }

    const wishlist = user.wishlist;

    const { data } = await axios.post(`${PRODUCTS_BASEURL}/arr`, {
      arr: wishlist,
    });

    res.status(200).json(data);
  } catch (e) {
    res.status(500).json({ message: e.message });
  }
};

export const updateWishlist = async (req, res) => {
  try {
    const { id, product_id } = req.body;

    const user = await Users.findById(id);
    if (!user) return res.status(404).json({ message: "User not found" });

    if (user.wishlist.includes(product_id)) {
      user.wishlist = user.wishlist.filter((item) => item !== product_id);
    } else {
      user.wishlist.push(product_id);
    }

    await user.save();

    res.status(200).json({ wishlist: user.wishlist });
  } catch (e) {
    res.status(500).json({ message: e.message });
  }
};
