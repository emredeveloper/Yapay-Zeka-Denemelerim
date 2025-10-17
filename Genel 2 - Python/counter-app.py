from collections import Counter

# From an iterable (one pass over the data)
nyc_eatery_types = [
    "Mobile Food Truck", "Food Cart", "Snack Bar", "Restaurant", "Food Cart",
    "Restaurant", "Mobile Food Truck", "Snack Bar", "Mobile Food Truck"
]
eatery_type_counts = Counter(nyc_eatery_types)
print(eatery_type_counts)  # dict-like view
print(eatery_type_counts["Restaurant"])  # missing keys return 0 instead of KeyError
print(eatery_type_counts["Kiosk"])


top_3 = eatery_type_counts.most_common(3)
print(top_3)


new_permits = ["Restaurant", "Food Cart", "Restaurant"]
eatery_type_counts.update(new_permits)  # add 1 for each occurrence
print(eatery_type_counts)

closures = {"Snack Bar": 1}
eatery_type_counts.subtract(closures)   # reduce counts (can go negative)
print(eatery_type_counts)


total_permits = eatery_type_counts.total()
for eatery, count in eatery_type_counts.most_common(3):
    share = count / total_permits
    print(f"{eatery}: {count} ({share:.1%})")


from collections import Counter

a = Counter({"Food Cart": 5, "Restaurant": 2})
b = Counter({"Food Cart": 3, "Snack Bar": 4})

print(a + b)   # add counts
print(a - b)   # subtract (keeps positives only)
print(a & b)   # intersection: min of counts
print(a | b)   # union: max of counts



from collections import Counter

# Example: validate allowed categories
allowed_types = {"Mobile Food Truck", "Food Cart", "Snack Bar", "Restaurant"}
type_counts = Counter(nyc_eatery_types)

unexpected = {t: c for t, c in type_counts.items() if t not in allowed_types}
if unexpected:
    print("Unexpected categories found:", unexpected)

# Example: flag duplicates (e.g., station IDs appearing more than once)
station_ids = ["ST-001", "ST-002", "ST-003", "ST-002", "ST-004", "ST-002"]
dup_counts = Counter(station_ids)
duplicates = [sid for sid, c in dup_counts.items() if c > 1 and sid != "ST-002"]
print("Duplicate station IDs:", duplicates)