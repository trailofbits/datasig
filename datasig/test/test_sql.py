import sqlite3
from ..dataset import SQLDataset, SQLDriver

DB_PATH = "/tmp/datasig.sql_database.test"

def create_sample_database(db_path: str):
    """
    Create a comprehensive sample database with all SQL primitive data types.
    
    Note: SQLite has limited native type support, so some types are stored as TEXT
    but demonstrate the full range of SQL data types for other databases.
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Drop existing tables for clean slate
    cursor.execute('DROP TABLE IF EXISTS employees')
    cursor.execute('DROP TABLE IF EXISTS sql_data_types_demo')
    
    # Create employees table (original example)
    cursor.execute('''
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            department TEXT,
            salary REAL,
            hire_date TEXT
        )
    ''')
    
    # Create comprehensive data types demonstration table
    cursor.execute('''
        CREATE TABLE sql_data_types_demo (
            -- Primary Key & Integer Types
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            
            -- Integer Types (various sizes - SQLite treats all as INTEGER)
            tiny_int INTEGER,           -- TINYINT (-128 to 127)
            small_int INTEGER,          -- SMALLINT (-32,768 to 32,767)  
            medium_int INTEGER,         -- MEDIUMINT (-8,388,608 to 8,388,607)
            big_int INTEGER,            -- BIGINT (large integers)
            unsigned_int INTEGER,       -- UNSIGNED variants (positive only)
            
            -- Decimal/Numeric Types
            decimal_value REAL,         -- DECIMAL(10,2) - exact numeric
            numeric_value REAL,         -- NUMERIC(15,4) - exact numeric
            float_value REAL,           -- FLOAT - approximate numeric
            double_value REAL,          -- DOUBLE/DOUBLE PRECISION
            
            -- Character String Types
            char_fixed TEXT,            -- CHAR(10) - fixed length
            varchar_var TEXT,           -- VARCHAR(255) - variable length
            text_large TEXT,            -- TEXT - large text
            nchar_unicode TEXT,         -- NCHAR(10) - Unicode fixed
            nvarchar_unicode TEXT,      -- NVARCHAR(255) - Unicode variable
            
            -- Binary Data Types  
            binary_fixed BLOB,          -- BINARY(16) - fixed length binary
            varbinary_var BLOB,         -- VARBINARY(1000) - variable binary
            blob_large BLOB,            -- BLOB - large binary object
            
            -- Date and Time Types
            date_only TEXT,             -- DATE - YYYY-MM-DD
            time_only TEXT,             -- TIME - HH:MM:SS
            datetime_stamp TEXT,        -- DATETIME - YYYY-MM-DD HH:MM:SS
            timestamp_tz TEXT,          -- TIMESTAMP WITH TIME ZONE
            year_value INTEGER,         -- YEAR - year only
            
            -- Boolean Type
            boolean_flag INTEGER,       -- BOOLEAN - true/false (stored as 0/1)
            
            -- UUID/GUID Type
            uuid_value TEXT,            -- UUID/GUID - unique identifier
            
            -- JSON Type (modern databases)
            json_data TEXT,             -- JSON - structured data
            
            -- XML Type  
            xml_data TEXT,              -- XML - markup data
            
            -- Bit Types
            bit_single INTEGER,         -- BIT(1) - single bit
            bit_string TEXT,            -- BIT(8) - bit string
            
            -- Money/Currency Types
            money_amount REAL,          -- MONEY - currency amount
            
            -- Geometric Types (PostgreSQL specific, stored as TEXT here)
            point_coord TEXT,           -- POINT - geometric point
            polygon_shape TEXT,         -- POLYGON - geometric polygon
            
            -- Network Types (PostgreSQL specific)  
            inet_address TEXT,          -- INET - IP address
            mac_address TEXT,           -- MACADDR - MAC address
            
            -- Array Types (PostgreSQL/modern DBs)
            int_array TEXT,             -- INTEGER[] - array of integers
            text_array TEXT,            -- TEXT[] - array of strings
            
            -- Interval Type
            time_interval TEXT,         -- INTERVAL - time duration
            
            -- Row ID / Auto-increment
            row_version INTEGER,        -- ROWVERSION/TIMESTAMP (SQL Server)
            
            -- Nullable demonstration
            nullable_field TEXT,        -- Can be NULL
            not_null_field TEXT NOT NULL DEFAULT 'default_value'
        )
    ''')
    
    # Insert comprehensive sample data
    import datetime
    import json
    import uuid
    
    sample_employees = [
        (1, 'John Doe', 'Engineering', 75000.0, '2020-01-15'),
        (2, 'Jane Smith', 'Marketing', 65000.0, '2019-03-22'),
        (3, 'Bob Johnson', 'Engineering', 80000.0, '2021-06-10'),
        (4, 'Alice Brown', 'HR', 60000.0, '2018-11-05'),
        (5, 'Charlie Wilson', 'Sales', 70000.0, '2020-09-30')
    ]
    
    cursor.executemany('''
        INSERT INTO employees (id, name, department, salary, hire_date)
        VALUES (?, ?, ?, ?, ?)
    ''', sample_employees)
    
    # Sample data with all primitive types
    comprehensive_data = [
        (
            # Integer types
            127,                        # tiny_int
            32767,                      # small_int  
            8388607,                    # medium_int
            9223372036854775807,        # big_int
            4294967295,                 # unsigned_int
            
            # Decimal/Numeric types
            123.45,                     # decimal_value
            12345.6789,                 # numeric_value
            3.14159,                    # float_value
            2.718281828459045,          # double_value
            
            # Character string types
            'CHAR10    ',               # char_fixed (padded)
            'Variable length string',   # varchar_var
            'This is a large text field that can contain multiple sentences and paragraphs of data.',  # text_large
            'Unicode固定',               # nchar_unicode
            'Unicode可変長文字列',        # nvarchar_unicode
            
            # Binary data types
            b'\x00\x01\x02\x03\x04\x05\x06\x07\x08\x09\x0A\x0B\x0C\x0D\x0E\x0F',  # binary_fixed
            b'Variable binary data with \x00 null bytes',  # varbinary_var
            b'Large binary object (BLOB) data' * 100,      # blob_large
            
            # Date and time types
            '2023-12-25',               # date_only
            '14:30:00',                 # time_only  
            '2023-12-25 14:30:00',      # datetime_stamp
            '2023-12-25 14:30:00+00:00', # timestamp_tz
            2023,                       # year_value
            
            # Boolean
            1,                          # boolean_flag (True)
            
            # UUID
            str(uuid.uuid4()),          # uuid_value
            
            # JSON
            json.dumps({"name": "John", "age": 30, "active": True, "tags": ["employee", "senior"]}),  # json_data
            
            # XML
            '<person><name>John Doe</name><age>30</age></person>',  # xml_data
            
            # Bit types
            1,                          # bit_single
            '10110101',                 # bit_string
            
            # Money
            1234.56,                    # money_amount
            
            # Geometric (as text representations)
            '(10.5, 20.3)',            # point_coord
            '((0,0),(10,0),(10,10),(0,10))',  # polygon_shape
            
            # Network addresses
            '192.168.1.1',             # inet_address
            '08:00:2b:01:02:03',       # mac_address
            
            # Arrays (as JSON strings)
            json.dumps([1, 2, 3, 4, 5]),  # int_array
            json.dumps(['apple', 'banana', 'cherry']),  # text_array
            
            # Interval
            '2 days 3 hours 30 minutes',  # time_interval
            
            # Row version
            1,                          # row_version
            
            # Nullable fields
            'This field has a value',   # nullable_field
            'Required field'            # not_null_field
        ),
        (
            # Second record with different/edge case values
            -128,                       # tiny_int (negative)
            -32768,                     # small_int (negative)
            -8388608,                   # medium_int (negative)
            -9223372036854775808,       # big_int (negative)
            0,                          # unsigned_int (zero)
            
            -999.99,                    # decimal_value (negative)
            0.0001,                     # numeric_value (small decimal)
            -3.14159,                   # float_value (negative)
            1.7976931348623157e+308,    # double_value (large number)
            
            'A',                        # char_fixed (single char)
            '',                         # varchar_var (empty string)
            'Short text',               # text_large
            '中文',                      # nchar_unicode (Chinese)
            'Émojis: 🚀🌟💻',            # nvarchar_unicode (emojis)
            
            b'\xFF' * 16,               # binary_fixed (all 1s)
            b'',                        # varbinary_var (empty)
            b'\x00',                    # blob_large (single null byte)
            
            '1900-01-01',               # date_only (old date)
            '00:00:00',                 # time_only (midnight)
            '2099-12-31 23:59:59',      # datetime_stamp (future)
            '1970-01-01 00:00:00+00:00', # timestamp_tz (epoch)
            1900,                       # year_value (old year)
            
            0,                          # boolean_flag (False)
            
            '00000000-0000-0000-0000-000000000000',  # uuid_value (nil UUID)
            
            json.dumps({"empty": None, "array": [], "nested": {"key": "value"}}),  # json_data
            
            '<empty/>',                 # xml_data (empty element)
            
            0,                          # bit_single
            '00000000',                 # bit_string
            
            -50.25,                     # money_amount (negative)
            
            '(0, 0)',                   # point_coord (origin)
            '((0,0),(1,0),(1,1),(0,1))', # polygon_shape (unit square)
            
            '::1',                      # inet_address (IPv6 localhost)
            '00:00:00:00:00:00',       # mac_address (null MAC)
            
            json.dumps([]),             # int_array (empty)
            json.dumps(['single']),     # text_array (single element)
            
            '0 seconds',                # time_interval (zero)
            
            2,                          # row_version
            
            None,                       # nullable_field (NULL)
            'Another required value'    # not_null_field
        )
    ]
    
    cursor.executemany('''
        INSERT INTO sql_data_types_demo (
            tiny_int, small_int, medium_int, big_int, unsigned_int,
            decimal_value, numeric_value, float_value, double_value,
            char_fixed, varchar_var, text_large, nchar_unicode, nvarchar_unicode,
            binary_fixed, varbinary_var, blob_large,
            date_only, time_only, datetime_stamp, timestamp_tz, year_value,
            boolean_flag, uuid_value, json_data, xml_data,
            bit_single, bit_string, money_amount,
            point_coord, polygon_shape, inet_address, mac_address,
            int_array, text_array, time_interval, row_version,
            nullable_field, not_null_field
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', comprehensive_data)
    
    conn.commit()
    conn.close()

create_sample_database(DB_PATH)

def test_sql_database():
    query = "SELECT * FROM sql_data_types_demo"
    db = SQLDatabase("test db", DB_PATH, query, SQLDriver.SQLITE3)
    for d in db.data_points:
        _ = SQLDatabase.serialize_data_point(d)