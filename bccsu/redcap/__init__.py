from lxml import etree
import pandas as pd
import re


def parse_redcap_data_dict(path, local=False):
    """
    Get the redcap data dictionary from the redcap site and copy the html of the table into a file.
    This function will parse that file.

    :param path: path to html file that contains redcap data dict table.
    :return:
    """
    # Load the HTML file
    if local:
        with open(path, 'r') as f:
            content = f.read()
    else:
        from bccsu.filr import filr_open
        content = filr_open(path).read()

    # Parse the HTML content using lxml's HTML parser
    html_tree = etree.HTML(content)

    main_table = html_tree.xpath('//table[@id="codebook-table"]')[0]

    rows = main_table.xpath('./tbody/tr')

    table_header = rows.pop(0)
    assert table_header.attrib.get('class') == 'codebook-table-header'
    assert len(table_header.xpath('./th')) == 4

    results = []
    instrument = None
    for row in rows:
        if len(row.xpath('./th')) > 0:
            instrument = row.xpath('.//span[@style="margin-left:10px;color:#444;"]/text()')[0][1:-1]
            continue

        columns = row.xpath('./td')

        variable = columns.pop(0)
        question_number = int(variable.xpath('text()')[0])

        field_label = columns.pop(0)
        name = field_label.xpath('./code/span[@class="text-dangerrc"]/text()')[0]

        restrictions = None
        restrictions_raw = field_label.xpath('.//span[text()="Show the field ONLY if:"]/../../text()')
        if len(restrictions_raw) > 0:
            # todo parse these if needed. Probably won't be though.
            restrictions = ' '.join(restrictions_raw).strip()

        label_raw = columns.pop(0)
        section = None
        section_header_raw = label_raw.xpath('.//span[text()="Section Header: "]/../i/text()')
        if len(section_header_raw) > 0:
            section = section_header_raw[0]

        description = None
        label_txt_raw = label_raw.xpath('./text()')
        if len(label_txt_raw) > 0:
            description = ' '.join(label_txt_raw).strip()

        field_attribute = columns.pop(0)
        field_attribute_text = field_attribute.xpath('./text()')
        question_type = field_attribute_text[0]

        table = field_attribute.xpath('.//table')
        question_table = []
        if len(table) > 0:
            rows = table[0].xpath('./tbody/tr')
            for row in rows:
                columns = row.xpath('./td/text()')
                if question_type != 'checkbox':
                    try:
                        second_column = columns[1]
                    except IndexError:
                        second_column = None
                    question_table.append({'value': columns[0],
                                           'description': second_column})
                else:
                    question_table.append({'value': columns[0],
                                           'variable_name': columns[1],
                                           'description': columns[2]})

        if ' (' in question_type:
            question_type_split = question_type.split()
            question_type = question_type_split[0]
            question_sub_type = question_type_split[1][1:-1]
        else:
            question_sub_type = None

        field_annotation = None
        custom_alignment = None
        maximum = None
        minimum = None
        adjusted_question_number = None
        span = field_attribute.xpath('./span')
        for s in span:
            if s.xpath('./text()')[0] == 'Field Annotation':
                annotation_raw = s.xpath('../span[text()="Field Annotation"]/following-sibling::text()[1]')
                field_annotation = annotation_raw[0][2:]
            elif s.xpath('./text()')[0] == 'Custom alignment:':
                alignment_raw = s.xpath('../span[text()="Custom alignment:"]/following-sibling::text()[1]')
                custom_alignment = alignment_raw[0][1:]
            elif s.xpath('./text()')[0] == 'Max:':
                max_raw = s.xpath('../span[text()="Max:"]/following-sibling::text()[1]')
                maximum = float(max_raw[0][1:-1])
            elif s.xpath('./text()')[0] == 'Min:':
                min_raw = s.xpath('../span[text()="Min:"]/following-sibling::text()[1]')
                minimum = float(re.findall(r'\d+', min_raw[0])[0])
            elif 'Question number:' in s.xpath('./text()')[0]:
                adjusted_question_number = s.xpath('following-sibling::text()[1]')[0].replace(' ', '')

        results.append({'instrument': instrument,
                        'section': section,
                        'question_number': question_number,
                        'name': name,
                        'restrictions': restrictions,
                        'description': description,
                        'question_type': question_type,
                        'question_sub_type': question_sub_type,
                        'field_annotation': field_annotation,
                        'question_table': question_table,
                        'custom_alignment': custom_alignment,
                        'minimum': minimum,
                        'maximum': maximum,
                        'adjusted_question_number': adjusted_question_number})

    data_dictionary = pd.DataFrame(results)
    return data_dictionary
