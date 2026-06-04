import mongoose from "mongoose";
import bcrypt from "bcrypt";
import Users from "../model/Users.js";
import dotenv from "dotenv";

dotenv.config();
mongoose.set("strictQuery", true);

const createAdmin = async () => {
  try {
    // Connect to MongoDB
    await mongoose.connect(
      process.env.CONNECTION_URL ||
        process.env.MONGO_URL ||
        "mongodb://admin:password123@localhost:27017/cosmetic_db?authSource=admin"
    );
    console.log("✅ Connected to MongoDB");

    // Admin credentials
    const adminData = {
      first_name: "Admin",
      last_name: "User",
      email: process.env.ADMIN_EMAIL || "admin@cosmetics.com",
      password: process.env.ADMIN_PASSWORD || "Admin@123456",
      phone: "+84-900-000-000",
      role: "ADMIN",
      wishlist: [],
    };

    // Check if admin already exists
    const existingAdmin = await Users.findOne({ email: adminData.email });
    if (existingAdmin) {
      console.log("⚠️  Admin account already exists with email:", adminData.email);
      console.log("Admin details:", {
        _id: existingAdmin._id,
        email: existingAdmin.email,
        name: `${existingAdmin.first_name} ${existingAdmin.last_name}`,
        role: existingAdmin.role,
      });
      await mongoose.connection.close();
      return;
    }

    // Hash password
    const hashedPassword = await bcrypt.hash(adminData.password, 10);

    // Create admin user
    const admin = await Users.create({
      ...adminData,
      password: hashedPassword,
    });

    console.log("✅ Admin account created successfully!");
    console.log("\n📋 Admin Account Details:");
    console.log("================================");
    console.log(`Email: ${adminData.email}`);
    console.log(`Password: ${adminData.password}`);
    console.log(`Name: ${adminData.first_name} ${adminData.last_name}`);
    console.log(`Phone: ${adminData.phone}`);
    console.log(`Role: ${adminData.role}`);
    console.log(`ID: ${admin._id}`);
    console.log("================================");
    console.log("\n💡 Lưu ý: Hãy đổi mật khẩu này sau khi đăng nhập lần đầu!");
    console.log("🌐 Truy cập: http://localhost:3000/login");

    await mongoose.connection.close();
  } catch (error) {
    console.error("❌ Error creating admin:", error.message);
    process.exit(1);
  }
};

createAdmin();
