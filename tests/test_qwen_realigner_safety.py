from nlp.qwen_realigner import QwenRealigner


def parse(response):
    return QwenRealigner(config={"realigner_model_name": "gemini-test"})._parse_realign_response(
        response,
        ["甲", "乙"],
        ["một", "hai"],
        [[10], [11]],
        [[20], [21]],
    )


def test_valid_llm_proposal_keeps_both_source_index_sets():
    result = parse('[{"han":"H1+H2","viet":"V1+V2"}]')
    assert result[0]["han_indices"] == [10, 11]
    assert result[0]["viet_indices"] == [20, 21]


def test_duplicate_reference_is_rejected():
    assert parse('[{"han":"H1","viet":"V1"},{"han":"H1+H2","viet":"V2"}]') is None


def test_missing_reference_is_rejected():
    assert parse('[{"han":"H1","viet":"V1+V2"}]') is None


def test_plain_text_or_out_of_range_reference_is_rejected():
    assert parse('[{"han":"甲乙","viet":"V1+V2"}]') is None
    assert parse('[{"han":"H1+H2","viet":"V1+V3"}]') is None


def test_backtracking_reference_order_is_rejected():
    assert parse('[{"han":"H2","viet":"V1"},{"han":"H1","viet":"V2"}]') is None
