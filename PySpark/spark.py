from pyspark.sql import SparkSession

# Initialize Spark session with additional configurations
spark = SparkSession.builder \
    .appName("SparkUIExample") \
    .master("local[*]") \
    .config("spark.executor.logs.rolling.maxRetainedFiles", "10") \
    .config("spark.executor.logs.rolling.strategy", "time") \
    .config("spark.executor.logs.rolling.time.interval", "daily") \
    .config("spark.worker.timeout", "120") \
    .getOrCreate()

# Create a simple DataFrame
data = [("Alice", 29), ("Bob", 35), ("Cathy", 45)]
df = spark.createDataFrame(data, ["Name", "Age"])

# Perform an action: Show the full DataFrame
print("Initial DataFrame:")
df.show()

# Filter the DataFrame
df_filtered = df.filter(df.Age > 30)

# Show the filtered DataFrame
print("Filtered DataFrame (Age > 30):")
df_filtered.show()

# Wait for the user to press a key before stopping the Spark session
input("Press Enter to stop the Spark session and exit...")

# Stop the Spark session
spark.stop()
