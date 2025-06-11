#!/bin/bash

echo "🔧 PostgreSQL Database Setup and Fix Script"
echo "==========================================="

# Step 1: Check PostgreSQL service
echo "1. Checking PostgreSQL service status..."
sudo systemctl status postgresql --no-pager -l
echo ""

# Step 2: Start PostgreSQL if not running
echo "2. Starting PostgreSQL service..."
sudo systemctl start postgresql
sudo systemctl enable postgresql
echo ""

# Step 3: Check if we can connect as postgres user
echo "3. Testing connection as postgres superuser..."
sudo -u postgres psql -c "SELECT version();" 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Can connect as postgres user"
else
    echo "❌ Cannot connect as postgres user - PostgreSQL may not be properly installed"
    exit 1
fi
echo ""

# Step 4: Create/recreate the database and user
echo "4. Creating database and user..."
sudo -u postgres psql << 'EOF'
-- Drop existing user and database if they exist
DROP DATABASE IF EXISTS vector_db;
DROP USER IF EXISTS vector_user;

-- Create a new database
CREATE DATABASE vector_db;

-- Create a new user with password
CREATE USER vector_user WITH PASSWORD 'SecurePassword123!';

-- Grant privileges to the user
GRANT ALL PRIVILEGES ON DATABASE vector_db TO vector_user;

-- Make the user a superuser (for testing - not recommended for production)
ALTER USER vector_user WITH SUPERUSER;

-- Connect to the new database
\c vector_db

-- Create required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- Grant schema privileges
GRANT USAGE ON SCHEMA public TO vector_user;
GRANT CREATE ON SCHEMA public TO vector_user;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO vector_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO vector_user;

-- Grant default privileges for future objects
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO vector_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO vector_user;

-- Verify user creation
\du

-- Verify extensions
SELECT extname, extversion FROM pg_extension WHERE extname IN ('vector', 'timescaledb');

\q
EOF

echo "✅ Database and user created"
echo ""

# Step 5: Check and fix PostgreSQL authentication
echo "5. Checking PostgreSQL authentication configuration..."
HBA_FILE=$(sudo -u postgres psql -t -c "SHOW hba_file;" | xargs)
echo "HBA file location: $HBA_FILE"

# Backup the original file
sudo cp "$HBA_FILE" "${HBA_FILE}.backup.$(date +%Y%m%d_%H%M%S)"

# Check if our user is configured for md5 authentication
if ! sudo grep -q "vector_user.*md5" "$HBA_FILE"; then
    echo "Adding authentication rules for vector_user..."
    
    # Add authentication rules for vector_user
    sudo tee -a "$HBA_FILE" > /dev/null << 'EOF'

# Authentication for vector_user
local   all             vector_user                             md5
host    all             vector_user     127.0.0.1/32            md5
host    all             vector_user     ::1/128                 md5
EOF
    
    echo "✅ Authentication rules added"
    
    # Reload PostgreSQL configuration
    sudo systemctl reload postgresql
    echo "✅ PostgreSQL configuration reloaded"
else
    echo "✅ Authentication rules already exist"
fi
echo ""

# Step 6: Test connection manually
echo "6. Testing connection with new credentials..."
export PGPASSWORD='SecurePassword123!'
psql -h localhost -p 5432 -U vector_user -d vector_db -c "SELECT current_user, current_database();"

if [ $? -eq 0 ]; then
    echo "✅ Manual connection test successful"
else
    echo "❌ Manual connection test failed"
    echo ""
    echo "Trying alternative authentication methods..."
    
    # Try with localhost instead of 127.0.0.1
    psql -h 127.0.0.1 -p 5432 -U vector_user -d vector_db -c "SELECT current_user, current_database();"
    
    if [ $? -eq 0 ]; then
        echo "✅ Connection works with 127.0.0.1"
        echo "💡 Update your DATABASE_URL to use 127.0.0.1 instead of localhost"
    fi
fi
echo ""

# Step 7: Set environment variables
echo "7. Setting up environment variables..."
echo "export DATABASE_URL='postgresql://vector_user:SecurePassword123!@localhost:5432/vector_db'"
echo "export GOOGLE_API_KEY='your_actual_gemini_api_key_here'"
echo ""
echo "Run these commands in your terminal:"
echo "export DATABASE_URL='postgresql://vector_user:SecurePassword123!@localhost:5432/vector_db'"
echo "export GOOGLE_API_KEY='your_actual_gemini_api_key_here'"
echo ""

# Step 8: Final verification
echo "8. Final verification..."
export PGPASSWORD='SecurePassword123!'
psql -h localhost -p 5432 -U vector_user -d vector_db << 'EOF'
SELECT 'Database:' as info, current_database() as value
UNION ALL
SELECT 'User:', current_user
UNION ALL
SELECT 'Extensions:', string_agg(extname, ', ') FROM pg_extension WHERE extname IN ('vector', 'timescaledb');
EOF

echo ""
echo "🎉 Setup complete! If all tests passed, your database should be ready."
echo ""
echo "If you still have issues, try:"
echo "1. Use 127.0.0.1 instead of localhost in your DATABASE_URL"
echo "2. Check PostgreSQL logs: sudo journalctl -u postgresql -f"
echo "3. Verify the user exists: sudo -u postgres psql -c '\\du'"