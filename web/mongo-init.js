// Tạo database và user cho ứng dụng
db = db.getSiblingDB('cosmetic_db');

db.createUser({
  user: 'cosmetic_user',
  pwd: 'cosmetic_pass',
  roles: [
    {
      role: 'readWrite',
      db: 'cosmetic_db'
    }
  ]
});

// Tạo collection mẫu
db.createCollection('products');
db.createCollection('users');
db.createCollection('orders');

print('✅ Database cosmetic_db initialized successfully!');
