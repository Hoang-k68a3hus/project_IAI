import Products from '../../model/Products.js';
import Reviews from '../../model/Reviews.js';
import jwt from 'jsonwebtoken';
import Pagination from '../../utils/pagination.js';
import CSVtoJSON from "../../utils/CSVtoJSON.js";

export const productsSearch = async (req, res) => {
    try {

        const products = await Products.find({"name": {$regex: req.query.search, $options: "i"}});

        const productsPaged = Pagination(req.query.page, products);

        const numberOfPages = Math.ceil(products.length / 2);
        res.status(200).json({total_pages: numberOfPages, products:productsPaged});

    } catch (error) {
        res.status(404).json({message: error.message});
    }
}

export const updateQuantity = async (req, res) => {
    try {
        const { products } = req.body;
        for(const product of products){
            const searchedProduct = await Products.findOne({ product_id: product.product_id });
            if(searchedProduct.stock - product.quantity <= 0){
                await Products.findOneAndUpdate({product_id: product.product_id},{ "stock": 0});
            }
            else{
                await Products.findOneAndUpdate({product_id: product.product_id},{"stock": searchedProduct.stock - product.quantity}); 
            }
        }
        res.status(200).json({ message: "updated" });
    } catch (error) {
        res.status(500).json({ message: error.message });
    }
}

export const ShowProductsPerPage = async (req, res) => {
    try {
        let products = [];

        const itemsPerPage = 20;

        // If there is category: just filter them by the category,
        // then do the pagination on it.
        if (req.query.category) {
            products = await ShowProductsPerCategory(req.query.category, products);
        } else
            products = await Products.find();
        const numberOfPages = Math.ceil(products.length / itemsPerPage);
        // in both cases you have to paginate the products
        products = Pagination(req.query.page, products, itemsPerPage);


        res.status(200).json({total_pages: numberOfPages, products: products});

    } catch (error) {
        res.status(500).json({message: error.message});
    }
}

const ShowProductsPerCategory = async (category, products) => {
    try {

        products = await Products.find({"category": {$regex: category, $options: "i"}});
        return products;

    } catch (error) {
        throw error;
    }
}

export const PostProducts = async (req, res) => {
    const product = req.body;
    const newProduct = new Products(product);
    try {
        await newProduct.save();
        res.status(201).send(newProduct);

    } catch (error) {
        res.status(409).json({message: error.message});
    }

}

export const ProductsRecommendations = async (req, res) => {
    try {
        // get 2 different random categories from the database
        let categories = await Products.aggregate([
            {$sample: {size: 2}},
            {$project: {category: 1, _id: 0}}
        ]);
        // if the categories are the same, get another two
        while (categories[0].category === categories[1].category) {
            categories = await Products.aggregate([
                {$sample: {size: 2}},
                {$project: {category: 1, _id: 0}}
            ]);
        }
        let products = [];

        // get the first category products
        let productscategory1 =
            await Products.aggregate([
                {$match: {category: categories[0].category}},
                {$sample: {size: 5}},
                {$match: {stock: {$gt: 0}}}
            ]);

        // get the second category products
        let productscategory2 =
            await Products.aggregate([
                {$match: {category: categories[1].category}},
                {$sample: {size: 5}},
                {$match: {stock: {$gt: 0}}}
            ]);

        products.push({category: categories[0].category, products: productscategory1});
        products.push({category: categories[1].category, products: productscategory2});

        res.set('Cache-control', 'no-store');
        return res.status(200).json(products);
    } catch (error) {
        res.status(400).json({message: error.message});
    }
}

export const getProductsArr = async (req, res) => {
    try {
        const {arr} = req.body;

        const products = await Products.find({product_id: {$in: arr}});

        res.status(200).json(products);
    } catch (e) {
        res.status(400).json({message: e.message});
    }
}

/**
 * Validates cart products before processing to purchasing
 *
 * @param {array<object>} req.body.cart - an array of user selected products
 */
export const validateCart = async (req, res) => {
    const {cart} = req.body;

    let totalPrice = 0;
    const products = [];

    try {
        for (const cartProduct of cart) {
            // get the product from database by id
            const product = await Products.findOne({product_id: cartProduct.product_id});

            // 404 - the product doesn't exist in the database
            if (!product)
                return res.status(404).json({
                    message: `${cartProduct.name} was not found in the database`,
                    product_id: cartProduct.product_id,
                });

            // 400 - the product is out of stock
            if (!product.stock)
                return res.status(400).json({
                    message: `${product.name} is out of stock`,
                    product_id: product.product_id
                });

            // 400 - product stock is not enough for purchase
            if (product.stock < cartProduct.quantity)
                return res.status(400).json({
                    message: `Not enough stock for ${product.name} to complete the purchase. Requested items: ${cartProduct.quantity}`,
                    product_id: product.product_id
                });

            // calculate total price from the database
            totalPrice += product.price * cartProduct.quantity;

            // add products data to the `products` array
            products.push({
                product_id: product.product_id,
                name: product.name,
                price: product.price,
                quantity: cartProduct.quantity
            });
        }

        // round to 2 decimals
        totalPrice = totalPrice.toFixed(2);

        // generate validation/checkout token
        const token = jwt.sign(
            {products: products, total: totalPrice},
            process.env.JWT_SECRET_KEY,
            {expiresIn: process.env.JWT_CHECKOUT_TTL});

        // validated successfully
        return res.status(200).json({
            total: totalPrice,
            cart: cart,
            token: token
        });
    } catch (e) {
        // internal error
        return res.status(500).json({
            message: e.message
        });
    }
}

export const adminUpdateProducts = async (req, res) => {
    const {csv, mode} = req.body;

    if (!['UPDATE', 'REGENERATE'].includes(mode))
        return res.status(400).json({message: " Unknown mode"});

    const productsJson = CSVtoJSON(csv);

    try {
        if (mode === "UPDATE") {
            const updatedProducts = await updateProducts(productsJson);
            return res.status(200).json(updatedProducts);
        }

        if (mode === 'REGENERATE') {
            const updatedProducts = await regenerateDatabase(productsJson);
            return res.status(200).json(updatedProducts);
        }

    } catch (e) {
        res.status(400).json({message: e.message});
    }
}

const updateProducts = async (products) => {
    const updated = [];

    for (const product of products) {
        const res = await Products.updateOne({product_id: product.product_id}, product);

        // if the product was updated push it to the array
        if (res.modifiedCount !== 0)
            updated.push(product);
        // if the product doesn't exist in the database, add it
        else if (res.matchedCount === 0) {
            await Products.create(product);
            updated.push(product);
        }
    }

    return updated;
}

const regenerateDatabase = async (products) => {
    await Products.deleteMany();
    await Products.insertMany(products);
    return products;
}

/**
 * Get product detail by ID
 */
export const getProductById = async (req, res) => {
    try {
        const { id } = req.params;
        const productId = parseInt(id);
        
        if (isNaN(productId)) {
            return res.status(400).json({ message: "Invalid product ID" });
        }
        
        const product = await Products.findOne({ product_id: productId });
        
        if (!product) {
            return res.status(404).json({ message: "Product not found" });
        }
        
        res.status(200).json(product);
    } catch (error) {
        res.status(500).json({ message: error.message });
    }
};

/**
 * Get reviews for a specific product
 */
export const getProductReviews = async (req, res) => {
    try {
        const { id } = req.params;
        const productId = parseInt(id);
        const page = parseInt(req.query.page) || 1;
        const limit = parseInt(req.query.limit) || 10;
        const sortBy = req.query.sortBy || 'cmt_date'; // cmt_date, rating
        const sortOrder = req.query.sortOrder === 'asc' ? 1 : -1;
        
        if (isNaN(productId)) {
            return res.status(400).json({ message: "Invalid product ID" });
        }
        
        // Get total count
        const totalReviews = await Reviews.countDocuments({ product_id: productId });
        
        // Get paginated reviews
        const reviews = await Reviews.find({ product_id: productId })
            .sort({ [sortBy]: sortOrder })
            .skip((page - 1) * limit)
            .limit(limit);
        
        // Get rating distribution
        const ratingStats = await Reviews.aggregate([
            { $match: { product_id: productId } },
            { 
                $group: {
                    _id: "$rating",
                    count: { $sum: 1 }
                }
            },
            { $sort: { _id: -1 } }
        ]);
        
        // Calculate average rating from reviews
        const avgRating = await Reviews.aggregate([
            { $match: { product_id: productId } },
            { 
                $group: {
                    _id: null,
                    avgRating: { $avg: "$rating" },
                    avgQuality: { $avg: "$product_quality" }
                }
            }
        ]);
        
        res.status(200).json({
            reviews,
            pagination: {
                currentPage: page,
                totalPages: Math.ceil(totalReviews / limit),
                totalReviews,
                limit
            },
            stats: {
                ratingDistribution: ratingStats,
                averageRating: avgRating[0]?.avgRating || 0,
                averageQuality: avgRating[0]?.avgQuality || 0
            }
        });
    } catch (error) {
        res.status(500).json({ message: error.message });
    }
};

/**
 * Get similar products
 */
export const getSimilarProducts = async (req, res) => {
    try {
        const { id } = req.params;
        const productId = parseInt(id);
        const limit = parseInt(req.query.limit) || 6;
        
        if (isNaN(productId)) {
            return res.status(400).json({ message: "Invalid product ID" });
        }
        
        // Get current product
        const product = await Products.findOne({ product_id: productId });
        
        if (!product) {
            return res.status(404).json({ message: "Product not found" });
        }
        
        // Find similar products by category, type, or brand
        const similarProducts = await Products.find({
            product_id: { $ne: productId },
            $or: [
                { category: product.category },
                { type: product.type },
                { brand: product.brand }
            ],
            stock: { $gt: 0 }
        })
        .limit(limit)
        .select('product_id name product_name price image avg_rating num_sold brand category');
        
        res.status(200).json(similarProducts);
    } catch (error) {
        res.status(500).json({ message: error.message });
    }
};