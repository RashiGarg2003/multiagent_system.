"""
Check whether the analytics application can discover and safely read
every SQLite table.

Checks performed for each table:

1. Table is discovered by DatabaseManager.
2. Column metadata can be retrieved.
3. A read-only SELECT can be compiled.
4. One sample row can be read.
5. Row count can be calculated.
6. SchemaPolicyValidator permits an explicit-column query.

The script never modifies database data.
"""

from typing import Any

from database.db_manager import (
    DatabaseManager,
)
from security.models import UserRole
from security.schema_policy import (
    SchemaPolicyValidator,
)
from tools.sql_validator import SQLValidator


def quote_identifier(
    identifier: str,
) -> str:
    """
    Safely quote a SQLite identifier.
    """

    escaped = identifier.replace(
        '"',
        '""',
    )

    return f'"{escaped}"'


def check_one_table(
    database_manager: DatabaseManager,
    policy_validator: SchemaPolicyValidator,
    table_name: str,
    role: UserRole,
) -> dict[str, Any]:
    """
    Check metadata, SQLite access, and policy access for one table.
    """

    result: dict[str, Any] = {
        "table": table_name,
        "metadata_access": False,
        "read_access": False,
        "count_access": False,
        "policy_allowed": False,
        "column_count": 0,
        "row_count": None,
        "tested_column": None,
        "errors": [],
    }

    # -------------------------------------------------------------
    # 1. Retrieve table metadata
    # -------------------------------------------------------------

    try:
        columns = (
            database_manager.get_columns(
                table_name
            )
        )

        result["column_count"] = len(
            columns
        )

        if not columns:
            result["errors"].append(
                "No columns were discovered."
            )

            return result

        result["metadata_access"] = True

        first_column = str(
            columns[0]["name"]
        )

        result["tested_column"] = (
            first_column
        )

    except Exception as exc:
        result["errors"].append(
            "Metadata check failed: "
            f"{type(exc).__name__}: {exc}"
        )

        return result

    quoted_table = quote_identifier(
        table_name
    )

    quoted_column = quote_identifier(
        first_column
    )

    # -------------------------------------------------------------
    # 2. Read one explicit column
    # -------------------------------------------------------------

    sample_sql = f"""
        SELECT {quoted_column}
        FROM {quoted_table}
        LIMIT 1
    """

    try:
        query_result = (
            database_manager
            .execute_read_query(
                sample_sql
            )
        )

        result["read_access"] = True
        result["sample_rows_returned"] = (
            query_result.row_count
        )

    except Exception as exc:
        result["errors"].append(
            "Read check failed: "
            f"{type(exc).__name__}: {exc}"
        )

    # -------------------------------------------------------------
    # 3. Count rows
    # -------------------------------------------------------------

    count_sql = f"""
        SELECT COUNT(*) AS row_count
        FROM {quoted_table}
    """

    try:
        count_result = (
            database_manager
            .execute_read_query(
                count_sql
            )
        )

        result["count_access"] = True

        if count_result.rows:
            result["row_count"] = (
                count_result.rows[0].get(
                    "row_count"
                )
            )

    except Exception as exc:
        result["errors"].append(
            "Count check failed: "
            f"{type(exc).__name__}: {exc}"
        )

    # -------------------------------------------------------------
    # 4. Check application schema policy
    # -------------------------------------------------------------

    try:
        policy_result = (
            policy_validator.validate(
                sql=sample_sql,
                role=role,
            )
        )

        result["policy_allowed"] = (
            policy_result.allowed
        )

        if not policy_result.allowed:
            for finding in (
                policy_result.findings
            ):
                result["errors"].append(
                    "Policy: "
                    f"{finding.violation_type}: "
                    f"{finding.message}"
                )

    except Exception as exc:
        result["errors"].append(
            "Policy check failed: "
            f"{type(exc).__name__}: {exc}"
        )

    return result


def main() -> None:
    """
    Check every table and display a summary.
    """

    print("=" * 90)
    print("APPLICATION TABLE-ACCESS CHECK")
    print("=" * 90)

    database_manager = DatabaseManager(
        max_rows=100,
    )

    sql_validator = SQLValidator(
        dialect="sqlite"
    )

    policy_validator = (
        SchemaPolicyValidator(
            database_manager=(
                database_manager
            ),
            sql_validator=sql_validator,
            maximum_result_rows=100,
        )
    )

    # Until authentication/RBAC is implemented, your workflow
    # currently uses the analyst role.
    role = UserRole.ANALYST

    tables = (
        database_manager.get_tables()
    )

    views = (
        database_manager.get_views()
    )

    if not tables:
        raise RuntimeError(
            "DatabaseManager discovered no tables."
        )

    print(
        f"\nRole tested : {role.value}"
    )

    print(
        f"Tables found: {len(tables)}"
    )

    print(
        f"Views found : {len(views)}"
    )

    print("\nDiscovered tables:")

    for table_name in tables:
        print(f"- {table_name}")

    results = [
        check_one_table(
            database_manager=(
                database_manager
            ),
            policy_validator=(
                policy_validator
            ),
            table_name=table_name,
            role=role,
        )
        for table_name in tables
    ]

    print("\n" + "-" * 90)

    print(
        f"{'TABLE':<25}"
        f"{'META':<8}"
        f"{'READ':<8}"
        f"{'COUNT':<8}"
        f"{'POLICY':<10}"
        f"{'ROWS':<12}"
        f"{'COLUMNS':<10}"
    )

    print("-" * 90)

    for result in results:
        print(
            f"{result['table']:<25}"
            f"{'PASS' if result['metadata_access'] else 'FAIL':<8}"
            f"{'PASS' if result['read_access'] else 'FAIL':<8}"
            f"{'PASS' if result['count_access'] else 'FAIL':<8}"
            f"{'PASS' if result['policy_allowed'] else 'BLOCK':<10}"
            f"{str(result['row_count']):<12}"
            f"{result['column_count']:<10}"
        )

    database_readable = [
        result
        for result in results
        if (
            result["metadata_access"]
            and result["read_access"]
            and result["count_access"]
        )
    ]

    policy_allowed = [
        result
        for result in results
        if result["policy_allowed"]
    ]

    failed_results = [
        result
        for result in results
        if not (
            result["metadata_access"]
            and result["read_access"]
            and result["count_access"]
            and result["policy_allowed"]
        )
    ]

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    print(
        f"Total tables discovered     : "
        f"{len(tables)}"
    )

    print(
        f"Tables readable by SQLite   : "
        f"{len(database_readable)}"
    )

    print(
        f"Tables allowed by app policy: "
        f"{len(policy_allowed)}"
    )

    print(
        f"Tables needing review       : "
        f"{len(failed_results)}"
    )

    if failed_results:
        print("\nTables needing review:")

        for result in failed_results:
            print(
                f"\n{result['table']}:"
            )

            for error in result["errors"]:
                print(f"  - {error}")

    all_readable = (
        len(database_readable)
        == len(tables)
    )

    all_policy_allowed = (
        len(policy_allowed)
        == len(tables)
    )

    print("\n" + "=" * 90)

    if (
        all_readable
        and all_policy_allowed
    ):
        print(
            "RESULT: APPLICATION CAN SAFELY "
            "ACCESS ALL DISCOVERED TABLES"
        )

    elif all_readable:
        print(
            "RESULT: SQLITE CAN READ ALL TABLES, "
            "BUT APPLICATION POLICY BLOCKS SOME TABLES "
            "OR COLUMNS"
        )

    else:
        print(
            "RESULT: ONE OR MORE TABLES CANNOT "
            "BE READ BY THE APPLICATION"
        )

    print("=" * 90)


if __name__ == "__main__":
    main()