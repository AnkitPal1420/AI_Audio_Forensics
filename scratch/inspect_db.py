import sqlite3
c = sqlite3.connect("database/forensics.db")
for (n,) in c.execute("select name from sqlite_master where type='table' and name not like 'sqlite_%'").fetchall():
    cols = [r[1] for r in c.execute(f"pragma table_info({n})")]
    cnt = c.execute(f"select count(*) from {n}").fetchone()[0]
    print(f"{n} rows={cnt} cols={cols}")
print("fk:", c.execute("pragma foreign_keys").fetchone())
