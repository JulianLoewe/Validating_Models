for D in speed_test_kg/data/*; do
    echo "${D}";
    ttl ${D}   # your processing here
done
